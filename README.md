# qwen-native.py Design Document

> A single-file, zero-dependency Python CLI Agent that connects to Qwen/DashScope and other LLM providers via the OpenAI-compatible protocol.

>
> Based on [claude-code-sdk](https://github.com/SeifBenayed/claude-code-sdk) by [SeifBenayed](https://github.com/SeifBenayed). The original project implemented the core Claude Code CLI functionality as single-file, zero-dependency scripts. `qwen-native.py` rewrites the API layer from Anthropic to OpenAI-compatible protocol, and adds Qwen OAuth, multi-Agent support, interactive UI, and more.
>
---

## 1. Overview

### 1.1 What It Is

`qwen-native.py` is a **~2750-line single-file Python CLI** with zero third-party pip dependencies — built entirely on the Python standard library. It delivers a terminal Agent experience similar to Claude Code / Qwen Code:

- Interactive REPL with tool-use loop
- Qwen OAuth subscription login (free tier: 1,000 requests/day)
- Multi-Agent parallel execution
- Streaming output with real-time spinner animation
- Slash command popup with live filtering (with scrolling & CJK wide-character support)

### 1.2 Design Goals

| Goal | Implementation |
|---|---|
| Zero dependencies | Only uses `urllib`, `json`, `threading`, `subprocess`, `unicodedata`, etc. from the standard library |
| Single file | Everything in one `.py` file — just run `python qwen-native.py` |
| Multi-provider | Unified OpenAI Chat Completions protocol, supports DashScope/OpenAI/DeepSeek/OpenRouter/etc. |
| No API key required | Qwen OAuth Device Flow login with automatic token management |
| Claude Code experience | Agent loop, tool use, multi-Agent parallelism, NDJSON Bridge protocol |
| Cross-platform | Windows (`msvcrt`) / macOS / Linux (`termios`) raw terminal input |

### 1.3 Relationship with claude-native.py

`qwen-native.py` is a rewrite of `claude-native.py` with these key differences:

| Dimension | claude-native.py | qwen-native.py |
|---|---|---|
| API protocol | Anthropic Messages API | OpenAI Chat Completions API |
| Authentication | Anthropic OAuth / API Key | Qwen OAuth Device Flow / API Key |
| Tool format | `{name, input_schema}` | Converted to `{type:"function", function:{name, parameters}}` |
| Message format | Content blocks | Role-based (system/user/assistant/tool) |
| Thinking | `thinking` content block | `reasoning_content` / `reasoning` field |
| Multi-Agent | None | Supported (Agent tool + ThreadPoolExecutor) |
| Interactive UI | Basic `input()` | SmartInput (raw terminal + live suggestion popup + CJK support) |
| Tool count | 5 (Bash/Read/Write/Glob/Grep) | 7 (+ Edit + Agent) |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      qwen-native.py                         │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐              │
│  │ ArgParser │  │  OAuth   │  │ProviderPreset│              │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘              │
│       └──────────────┼───────────────┘                      │
│                      ▼                                      │
│              ┌───────────────┐                              │
│              │   main()      │ ← Entry: parse, auth, route  │
│              └───┬───┬───┬───┘                              │
│          ┌───────┘   │   └────────┐                         │
│          ▼           ▼            ▼                          │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐                   │
│  │Interactive│ │ One-shot  │ │  NDJSON  │ ← Three modes      │
│  │   Mode   │ │   Mode    │ │  Bridge  │                    │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘                   │
│       └─────────────┼────────────┘                          │
│                     ▼                                       │
│             ┌───────────────┐                               │
│             │   AgentLoop   │ ← Core loop                    │
│             └───┬───────┬───┘                               │
│                 │       │                                   │
│          ┌──────┘       └──────┐                            │
│          ▼                     ▼                            │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │ OpenAIClient │    │ ToolRegistry │                       │
│  │ (SSE stream) │    │  (7 tools)   │                       │
│  └──────────────┘    └──────┬───────┘                       │
│                             │                               │
│       ┌──────┬───────┬──────┼────┬───────┬──────┐           │
│       ▼      ▼       ▼      ▼    ▼       ▼      ▼           │
│     Bash   Read   Write   Edit  Glob   Grep  Agent          │
│                                                  │          │
│                                          ┌───────┘          │
│                                          ▼                  │
│                                  ┌──────────────┐           │
│                                  │  Sub-Agent   │           │
│                                  │ (independent  │           │
│                                  │  session,     │           │
│                                  │  no nesting)  │           │
│                                  └──────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Module Inventory

| Module | Responsibility |
|---|---|
| Globals & Aliases | Logging, model aliases (qwen/gpt/deepseek/claude), provider presets (6) |
| ArgParser | CLI argument parsing, `_resolve_config` auto-detection of provider/key/model |
| Qwen OAuth | Device Flow login, PKCE, token refresh, `resource_url` dynamic endpoint, credential caching |
| HTTP Helpers | urllib-based HTTP requests and SSE streaming |
| OpenAIClient | API client, translates OpenAI SSE into Anthropic-style unified event stream |
| Format Converters | Bidirectional tool/message format conversion between internal and OpenAI formats |
| ToolRegistry | Tool registry with allow/deny list filtering, distinguishes built-in vs. external tools |
| Built-in Tools | 7 built-in tool definitions and executors (including Agent tool) |
| PromptBuilder | Builds system prompt, loads CLAUDE.md, appends parallel-tool instructions |
| AgentLoop | Core Agent loop: stream → parse → tool execution (with parallelism) → loop |
| SessionManager | JSONL session persistence (`~/.qwen-native/sessions/`) |
| NdjsonBridge | NDJSON protocol bridge mode with async external tool callbacks |
| UI: _C | ANSI color/style helper class |
| UI: Spinner | Background-thread-driven terminal spinner animation |
| UI: SmartInput | Cross-platform raw terminal input + live slash command popup + CJK-aware cursor |
| InteractiveMode | Interactive REPL: banner, input, tool display, thinking/text rendering, stats |
| Main | Entry point: auth priority resolution, mode dispatch |

---

## 3. Core Components

### 3.1 OpenAIClient — API Client

#### Purpose

Communicates with OpenAI-compatible Chat Completions APIs and translates OpenAI SSE streams into a unified internal event format.

#### Unified Event Protocol

`OpenAIClient.stream()` accepts an OpenAI-format request body and yields normalized events identical to the Anthropic SSE protocol:

```
OpenAI SSE chunk                        Normalized event
────────────────────────                ─────────────────
choices[0].delta.content           →    content_block_delta (text_delta)
choices[0].delta.reasoning_content →    content_block_delta (thinking_delta)
choices[0].delta.tool_calls        →    content_block_start (tool_use) +
                                        content_block_delta (input_json_delta)
choices[0].finish_reason           →    message_delta (stop_reason) +
                                        message_stop
usage                              →    message_start / message_delta (usage)
```

**Design rationale**: By completing protocol translation at the Client layer, `AgentLoop` code is completely agnostic to the underlying API differences. Both parent and child agents share the same loop logic.

#### finish_reason Mapping

```python
"stop"           → "end_turn"      # Normal completion
"tool_calls"     → "tool_use"      # Tool execution needed
"length"         → "max_tokens"    # Token limit reached
"content_filter" → "end_turn"      # Content filtered
```

#### Retry Strategy

- HTTP 429 (rate limit) and 529 (overload): exponential backoff, up to 3 attempts
- Intervals: 2s → 4s → 8s

#### Provider-Specific Headers

```python
# DashScope
headers["X-DashScope-CacheControl"] = "enable"
headers["X-DashScope-UserAgent"] = ua

# OpenRouter
headers["HTTP-Referer"] = "..."
headers["X-Title"] = "qwen-native"
```

### 3.2 Format Converters

#### Tool Definition Conversion (`tools_to_openai`)

```
Internal format (Anthropic-style)       OpenAI format
─────────────────────────               ──────────────────
{                                       {
  "name": "Bash",                         "type": "function",
  "description": "...",       →           "function": {
  "input_schema": {...}                     "name": "Bash",
}                                           "description": "...",
                                            "parameters": {...}
                                          }
                                        }
```

#### Message Conversion (`messages_to_openai`)

Converts internal content-block format to OpenAI's role-based format:

| Internal format | OpenAI format |
|---|---|
| `{"role": "user", "content": "text"}` | `{"role": "user", "content": "text"}` |
| `{"role": "user", "content": [{"type": "tool_result", ...}]}` | `{"role": "tool", "tool_call_id": ..., "content": ...}` |
| `{"role": "assistant", "content": [text + tool_use blocks]}` | `{"role": "assistant", "content": ..., "tool_calls": [...]}` |
| thinking block | `"reasoning_content"` field |

### 3.3 AgentLoop — Core Loop

#### Flow

```
User message → messages[]
     │
     ▼
┌─── while turn < maxTurns ─────────────────────────────┐
│                                                        │
│  1. messages_to_openai() → Convert to OpenAI format    │
│  2. tools_to_openai()   → Convert tool definitions     │
│  3. client.stream(body) → SSE streaming request        │
│  4. Parse events, assemble content_blocks[]            │
│  5. Append assistant reply to messages[]               │
│                                                        │
│  if stop_reason != "tool_use" → Return result, done    │
│                                                        │
│  6. Execute tools:                                     │
│     ├─ Single tool → Execute directly                  │
│     └─ Multiple tools → ThreadPoolExecutor parallel    │
│  7. Append tool_results as user message to messages[]  │
│                                                        │
│  └─── Back to step 1 ─────────────────────────────────┘
```

#### Parallel Execution Strategy

```python
run_parallel = len(tool_use_blocks) > 1

if run_parallel:
    with ThreadPoolExecutor(max_workers=len(tool_use_blocks)) as pool:
        # Execute all tool calls in parallel
else:
    # Execute single tool directly
```

**Design decision**: Whenever the model returns multiple `tool_use` blocks in a single turn, all tools are executed in parallel (including Agent and regular tools). The model placing multiple tools in one response implicitly asserts they are independent.

**Model guidance**: System prompt and Agent tool description include instructions to encourage the model to return multiple tool_calls in a single response:

```
# In system prompt:
"When you have multiple independent tasks, launch ALL Agent calls
 in one response so they run concurrently."

# In Agent tool description:
"IMPORTANT: When you have multiple independent tasks, launch multiple
 Agent calls in a SINGLE response to run them in parallel."
```

> **Note**: Actual parallelism depends on the model's compliance. Some models tend to return only one tool_call per turn, resulting in sequential execution. This is a model behavior limitation, not a code issue.

#### Thinking/Text Separated Rendering

Thinking (reasoning process) and text (final answer) in model output are rendered separately via callbacks:

```
on_thinking(delta):
  → Gray (DIM) style, prefixed with "💭 Thinking..."

on_text(delta):
  → If previously in thinking state, output RESET to close gray style
  → Normal style for the actual answer
```

This ensures thinking content appears in gray while the final answer renders in normal style.

#### Stream Event Processing

```python
for event_type, data in self.client.stream(body):
    match event_type:
        "message_start"       → Initialize usage
        "content_block_start" → Create new block (text/thinking/tool_use)
        "content_block_delta" → Incremental append + trigger callbacks
        "content_block_stop"  → Block complete, JSON parse tool_use input
        "message_delta"       → Get stop_reason and usage
        "message_stop"        → Message complete
```

### 3.4 Multi-Agent Architecture

#### Design Principles

```
Parent Agent (main session)
  │
  ├─ Agent(prompt="Search API endpoints")
  │    └─ Sub-Agent
  │         ├─ Independent messages[] (does not share parent history)
  │         ├─ Independent ToolRegistry (Bash/Read/Write/Edit/Glob/Grep, NO Agent)
  │         ├─ Independent system prompt + sub-agent role instructions
  │         ├─ Independent maxTurns=15
  │         └─ Returns text result → parent sees as tool_result
  │
  ├─ Agent(prompt="Run tests")      ← Runs in parallel (if model emits both)
  │    └─ Sub-Agent (same as above)
  │
  └─ Parent Agent synthesizes both results
```

#### Key Constraints

| Constraint | Reason |
|---|---|
| No nesting | Sub-agent ToolRegistry does not register Agent tool, preventing recursion |
| Independent session | Avoids token waste, protects parent context window from sub-agent intermediate noise |
| Result isolation | All sub-agent intermediate steps (tool calls, thinking) stay in sub-agent's messages only |
| Parallel execution | Multiple tool calls in the same turn run concurrently via ThreadPoolExecutor |
| UI visibility | Sub-agent tool activity shown via `on_agent_tool_use` callback (`↳` prefix) |

#### Information Flow

```
Parent → Child:  prompt (task description) + system prompt (with sub-agent role)
Child  → Parent: Single text result (all intermediate steps discarded)
```

#### Actual Parallel Behavior

```
Scenario A: Model returns multiple Agent calls at once (ideal)
──────────────────────────────────────────────────────────
Turn 1: model → [Agent("task A"), Agent("task B")]
         → ThreadPoolExecutor parallel
         → Thread 1: task A (independent AgentLoop)
         → Thread 2: task B (independent AgentLoop)  ← True parallelism
Turn 2: model ← [result A, result B] → Summary response

Scenario B: Model returns Agent calls one at a time (common with some models)
──────────────────────────────────────────────────────────
Turn 1: model → [Agent("task A")]
         → Execute task A
Turn 2: model ← [result A] → [Agent("task B")]
         → Execute task B                             ← Actually sequential
Turn 3: model ← [result B] → Summary response
```

System prompt guidance encourages Scenario A, but actual behavior depends on model capability.

#### Agent Tool Definition

```python
{
    "name": "Agent",
    "input_schema": {
        "properties": {
            "prompt":        # Required — complete task description
            "description":   # Optional — 3-5 word summary (for UI display and spinner)
            "model":         # Optional — model override (sub-agent can use a different model)
            "max_turns":     # Optional — max tool turns (default: 15)
            "system_prompt": # Optional — system prompt override
        }
    }
}
```

#### Parent vs. Sub-Agent Tool Availability

| | Parent Agent | Sub-Agent |
|---|---|---|
| Bash | ✓ | ✓ |
| Read | ✓ | ✓ |
| Write | ✓ | ✓ |
| Edit | ✓ | ✓ |
| Glob | ✓ | ✓ |
| Grep | ✓ | ✓ |
| **Agent** | **✓** | **✗** ← Prevents nesting |

### 3.5 ToolRegistry

#### Design

```python
class ToolRegistry:
    _tools: dict[str, {definition, executor}]
    _allowed: list | None      # Allow list (whitelist)
    _disallowed: list | None   # Deny list (blacklist)
```

- `executor = None` indicates an external tool (forwarded by NdjsonBridge)
- `get_definitions()` applies filtering before returning tool schemas
- `execute(name, input)` calls the executor and wraps the result
- Agent tool is registered separately via `register_agent_tool()` (requires client and cfg references)

#### Built-in Tools

| Tool | Function | Key Parameters |
|---|---|---|
| **Bash** | Execute shell commands | `command`, `timeout`(ms, default 120000, max 600000) |
| **Read** | Read files (with line numbers, line truncation at 2000 chars) | `file_path`, `offset`, `limit`(default 2000) |
| **Write** | Write files (auto-creates parent directories) | `file_path`, `content` |
| **Edit** | Exact string replacement (uniqueness check) | `file_path`, `old_string`, `new_string`, `replace_all` |
| **Glob** | File name pattern matching (skips hidden directories) | `pattern`, `path` |
| **Grep** | Content search (prefers rg, falls back to grep) | `pattern`, `path`, `glob`, `output_mode`, `-i`, `-C`/`-A`/`-B` |
| **Agent** | Launch independent sub-agent | `prompt`, `description`, `model`, `max_turns`, `system_prompt` |

---

## 4. Authentication System

### 4.1 Authentication Priority

```
1. --api-key CLI argument
2. Provider-specific environment variable (DASHSCOPE_API_KEY / OPENAI_API_KEY / ...)
3. OPENAI_API_KEY environment variable (universal fallback)
4. Qwen OAuth cached credentials (~/.qwen/oauth_creds.json)
5. In interactive mode, automatically trigger OAuth login flow
```

### 4.2 Qwen OAuth Device Flow (RFC 8628)

```
User                    qwen-native                chat.qwen.ai
  │                         │                           │
  │   python qwen-native.py --login                     │
  │                         │                           │
  │                         │──── POST /device/code ───→│
  │                         │←── device_code,           │
  │                         │    user_code,             │
  │                         │    verification_uri ──────│
  │                         │                           │
  │  ┌────────────────────────────────────────────┐     │
  │  │ Display:                                   │     │
  │  │   Visit https://chat.qwen.ai/authorize     │     │
  │  │   Your code: GTPFZRDF                      │     │
  │  └────────────────────────────────────────────┘     │
  │                         │                           │
  │   (User authorizes      │                           │
  │    in browser)          │                           │
  │                         │── poll POST /token ──────→│
  │                         │←── authorization_pending──│
  │                         │── poll POST /token ──────→│
  │                         │←── access_token,          │
  │                         │    refresh_token,         │
  │                         │    resource_url ──────────│
  │                         │                           │
  │                         │──→ Save to ~/.qwen/       │
  │                         │    oauth_creds.json       │
  │                         │                           │
  │   Login successful!     │                           │
```

#### Key Details

- **PKCE**: Uses S256 method (`code_verifier` + `code_challenge`)
- **resource_url**: The actual API endpoint returned by OAuth (e.g., `portal.qwen.ai`), not the default DashScope URL. `get_qwen_oauth_token()` returns a `(token, base_url)` tuple
- **Token refresh**: Auto-refreshes using `refresh_token` 30 seconds before expiry; clears credentials on refresh failure
- **WAF bypass**: Must include `User-Agent: QwenCode/1.0.0` header, otherwise Alibaba Cloud WAF returns HTML instead of JSON
- **Polling strategy**: Polls every 2 seconds; on 429, backs off automatically (interval ×1.5, max 10s)
- **Browser launch**: Windows uses `os.startfile()`, macOS uses `open`, Linux uses `xdg-open`

#### Credential File Format

```json
// ~/.qwen/oauth_creds.json
{
  "access_token": "de3Zdps...",
  "refresh_token": "phEBU1N...",
  "token_type": "Bearer",
  "expiry_date": 1774486436122,
  "resource_url": "portal.qwen.ai"
}
```

### 4.3 Provider Presets

| Provider | Base URL | Environment Variable | Default Model |
|---|---|---|---|
| dashscope | `dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `coder-model` |
| openai | `api.openai.com/v1` | `OPENAI_API_KEY` | `gpt-4o` |
| deepseek | `api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| openrouter | `openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `qwen/qwen3-coder` |
| modelscope | `api-inference.modelscope.cn/v1` | `MODELSCOPE_API_KEY` | `qwen3-coder` |

Provider auto-detection order: `--provider` flag → `--base-url` URL pattern → environment variable presence → default `dashscope`.

---

## 5. Interactive UI Design

### 5.1 SmartInput — Real-time Input Component

#### Architecture

```
┌─ SmartInput ──────────────────────────────────────────┐
│                                                        │
│  _getch()            ← Cross-platform raw key reading  │
│    ├─ Windows: msvcrt.getwch()                         │
│    └─ Unix: termios + tty.setraw()                     │
│                                                        │
│  _char_width(ch)     ← Single character terminal width │
│    └─ unicodedata.east_asian_width: W/F→2, others→1   │
│                                                        │
│  _display_width(s)   ← String terminal width (CJK)    │
│                                                        │
│  read(prompt)        ← Main input loop                 │
│    ├─ Key → modify text/cursor                         │
│    ├─ Update matches (slash command filtering)         │
│    └─ _render() redraw                                 │
│                                                        │
│  _render()           ← Terminal rendering              │
│    ├─ \r\033[K clear line + write prompt+text           │
│    ├─ \033[J clear everything below (remove old popup) │
│    ├─ Draw popup box with virtual scrolling window     │
│    └─ \033[nA return to prompt line + CJK-aware cursor │
│                                                        │
└────────────────────────────────────────────────────────┘
```

#### CJK Wide Character Handling

CJK characters occupy 2 terminal columns while ASCII occupies 1. Cursor positioning must use display columns, not character count:

```python
# Cursor positioning
text_cols = _display_width(text[:cursor])   # NOT len(text[:cursor])
target_col = prompt_cols + text_cols
```

Uses `unicodedata.east_asian_width()`: `W` (Wide) and `F` (Fullwidth) return 2, all others return 1.

#### Key Bindings

| Key | Without popup | With popup |
|---|---|---|
| Character | Insert into text | Insert + update filter |
| Backspace | Delete previous char | Delete + update filter |
| Tab | No action | Accept selected command |
| ↑ | Previous history entry | Move selection up |
| ↓ | Next history entry | Move selection down (with scroll) |
| Esc | No action | Close popup |
| Enter | Submit input | Accept selection + submit |
| Ctrl+C | Interrupt (double-tap to exit) | Close popup + interrupt |
| Ctrl+U | Clear text before cursor | Clear + update |
| Ctrl+D | Exit (on empty line) | — |
| Home/End | Move cursor to start/end | — |
| Left/Right | Move cursor left/right | — |
| Del | Delete char after cursor | — |

#### Popup Virtual Scrolling

When commands exceed 8 items, the popup supports virtual scrolling that keeps the selected item visible:

```
┌──────────────────────────────────────────────────────────┐
│  ▲ 2 more                                                │  ← Hidden items above
│  /clear          Start a new conversation                │
│  /model          Switch or show model                    │
│▶ /provider       Switch provider                         │  ← Blue background highlight
│  /thinking       Toggle extended thinking                │
│  /cost           Show token usage                        │
│  /session        Show current session info               │
│  /login          Login via Qwen OAuth                    │
│  /logout         Remove saved credentials                │
│  ▼ 2 more                                                │  ← Hidden items below
└──────────────────────────────────────────────────────────┘
  Tab to select · Esc to dismiss
```

The scroll window follows the `selected` index: when the selection moves beyond the visible range, the window scrolls automatically.

#### Terminal Rendering Strategy

Each `_render()` call follows these steps:

1. `\r\033[K` — Return to line start, clear entire line
2. Write prompt + text (text starting with `/` is cyan-colored)
3. `\033[J` — **Clear everything from cursor to end of screen** (key: removes old popup remnants)
4. If matches exist, draw new popup (each line via `\n`)
5. `\033[{popup_lines}A` — Return to prompt line
6. `\r\033[{target_col}C` — Position cursor precisely (CJK-aware column count)

`\033[J]` in step 3 atomically clears all old content, avoiding the complexity and scroll issues of line-by-line clearing with position counting.

### 5.2 Spinner — Wait Animation

```python
FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
```

- Driven by a daemon background thread, 80ms/frame
- Display format: `  ⠋ Thinking  12s · esc to cancel`
- Auto-updates text during tool execution: `⠹ Running Bash`, `⠸ Agent: searching…`
- `stop()` calls `\r\033[K` to clear the spinner line

### 5.3 Tool Call Display

```
  ⊷ Bash  ls -la                        ← Regular tool (cyan ⊷)
    ✓ (15 lines)                         ← Success (green ✓ + result summary)

  ◈ Agent  Search API endpoints          ← Agent tool (magenta ◈)
      ↳ Agent(Search API endpoints) → Grep ✓  ← Sub-agent activity (gray)
      ↳ Agent(Search API endpoints) → Read ✓
    ✓ (23 lines)                         ← Agent result

  ⊷ Read  /path/to/file.py
    ✓                                    ← Short result, no summary

  ⊷ Edit  /path/to/file.py
    ✕ Error: old_string not found        ← Failure (red ✕ + error preview)
```

Tool parameter summaries are displayed intelligently based on tool type:
- Bash → command text (truncated to 60 chars)
- Read/Write/Edit → file path
- Glob/Grep → search pattern
- Agent → description field (in magenta)

### 5.4 Startup Banner

```
  ╔═══════════════════════════════╗
  ║  ╔═╗ ╦ ╦╔═╗╔╗╔  ╔═╗╔═╗╔╦╗╔═╗║     ← Gradient colors (blue→cyan→magenta)
  ║  ║═╬╗║║║║╣ ║║║  ║  ║ ║ ║║║╣ ║
  ║  ╚═╝╚╚╩╝╚═╝╝╚╝  ╚═╝╚═╝═╩╝╚═╝║
  ╚═══════════════════════════════╝

  >_ qwen-native                         ← Bold
  dashscope | coder-model                 ← Gray (provider | model)
  ~/Projects/myapp                        ← Working directory (~ shortened, long paths truncated)

  ↻ Resumed session (15 messages)         ← Only shown when resuming (green)

  Tip: Type / for commands, Tab to autocomplete  ← Random tip
```

### 5.5 Status Icon System

| Icon | Meaning | Color |
|---|---|---|
| `✓` | Success | Green |
| `✕` | Error | Red |
| `△` | Warning / Interrupted | Yellow |
| `●` | Info | Gray |
| `⊷` | Regular tool executing | Cyan |
| `◈` | Agent executing | Magenta |
| `↳` | Sub-agent internal tool activity | Gray |
| `↻` | Session resumed | Green |
| `?` | Unknown command suggestion | Yellow |
| `💭` | Thinking | Gray |

### 5.6 Response Statistics

A gray statistics line is displayed after each response:

```
  3.2s · ↑1234 ↓567 · 2 tools
```

- Duration (seconds or min:sec format)
- Input/output token counts (if provided by the provider)
- Tool call count

---

## 6. Three Execution Modes

### 6.1 Interactive REPL

```bash
python qwen-native.py
```

Full interactive experience: banner, SmartInput, spinner, session persistence, slash commands.

### 6.2 One-shot

```bash
python qwen-native.py -p "explain this code"
```

Executes a single request and exits. Output goes to stdout, debug info to stderr. Suitable for pipelines and script integration.

### 6.3 NDJSON Bridge

```bash
echo '{"type":"message","content":"hi"}' | python qwen-native.py --ndjson
```

Communicates as a subprocess via stdin/stdout NDJSON protocol. Supports:

- External tool registration and async callbacks (via `threading.Event` blocking)
- Streaming text event forwarding
- Session management (set_model, interrupt, end_session, ping/pong)

---

## 7. Session Persistence

### 7.1 Storage Location

```
~/.qwen-native/sessions/{uuid}.jsonl
```

### 7.2 Format

One JSON object per line:

```jsonl
{"role": "user", "content": "hello"}
{"role": "assistant", "content": "Hi! How can I help?"}
{"role": "user", "content": "read file.py"}
{"role": "assistant", "content": [{"type": "tool_use", ...}]}
```

### 7.3 Resuming

```bash
python qwen-native.py --resume           # Resume most recent session
python qwen-native.py --session-id UUID   # Resume specific session
```

---

## 8. Data Flow

### 8.1 Complete Tool Call Interaction

```
User input: "What files are in the current directory?"
    │
    ▼
InteractiveMode._process_input()
    │
    ├─ messages.append(user message)
    ├─ Spinner.start("Thinking")
    │
    ▼
AgentLoop.run()
    │
    ├─ messages_to_openai()  → [{"role":"system",...}, {"role":"user","content":"..."}]
    ├─ tools_to_openai()     → [{"type":"function","function":{...}}, ...]
    │
    ├─ OpenAIClient.stream(body)
    │   ├─ POST /v1/chat/completions (stream=true)
    │   ├─ SSE: data: {"choices":[{"delta":{"tool_calls":[...]}}]}
    │   └─ _translate_openai_stream() → Normalized events
    │
    ├─ Parse events → content_blocks = [{"type":"tool_use","name":"Glob",...}]
    ├─ stop_reason = "tool_use"
    │
    ├─ on_tool_use callback → Spinner.stop() + display "⊷ Glob *"
    ├─ registry.execute("Glob", {"pattern":"*"})
    │   └─ _exec_glob() → Returns file list
    ├─ on_tool_result callback → display "✓ (5 lines)"
    │
    ├─ messages.append(tool_result)
    │
    ├─ Turn 2: client.stream(updated messages)
    │   └─ SSE: data: {"choices":[{"delta":{"content":"The directory has..."}}]}
    │
    ├─ on_text callback → Spinner.stop() + stream text output (normal style)
    ├─ stop_reason = "end_turn"
    │
    └─ return {"text": "...", "usage": {...}, "turns": 2}
    │
    ▼
InteractiveMode
    ├─ Display stats: "3.2s · ↑1234 ↓567 · 1 tool"
    └─ sessions.append(assistant message)
```

### 8.2 Multi-Agent Parallel Execution

```
AgentLoop Turn 1:
    model returns: [tool_use(Agent, "task A"), tool_use(Agent, "task B")]
    │
    ├─ len(tool_use_blocks) = 2 → run_parallel = true
    │
    └─ ThreadPoolExecutor(max_workers=2)
         │
         ├─ Thread 1: _exec_agent("task A")
         │   ├─ new ToolRegistry (no Agent tool)
         │   ├─ new messages = [{"role":"user","content":"task A"}]
         │   ├─ new AgentLoop.run()
         │   │   ├─ Turn 1: Bash("command A") → result
         │   │   └─ Turn 2: text response
         │   └─ return {"content": "task A result"}
         │
         └─ Thread 2: _exec_agent("task B")    ← True parallelism
             ├─ new ToolRegistry (no Agent tool)
             ├─ new messages = [{"role":"user","content":"task B"}]
             └─ ...same as above

AgentLoop Turn 2:
    messages = [..., tool_result("task A result"), tool_result("task B result")]
    model returns: text summarizing both results
```

### 8.3 Thinking + Text Rendering Flow

```
SSE event stream:
    reasoning_content: "This is simple math..."  → on_thinking → Gray DIM output
    reasoning_content: "1+1=2"                   → on_thinking → Gray DIM output
    content: "1+1 equals "                       → on_text → RESET closes gray + normal output
    content: "**2**"                             → on_text → Normal output
    finish_reason: "stop"                        → message_delta → end_turn

Terminal result:
    💭 Thinking...
    This is simple math...1+1=2               ← Gray (DIM)

    1+1 equals **2**                          ← Normal style (not gray)
```

---

## 9. Configuration

### 9.1 CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `-m, --model` | string | auto | Model name or alias |
| `-p, --print` | string | — | One-shot mode |
| `--provider` | string | auto | Provider preset |
| `--base-url` | string | auto | API URL |
| `--api-key` | string | env | API key |
| `--max-turns` | int | 25 | Max tool turns |
| `--max-tokens` | int | 16384 | Max output tokens |
| `--temperature` | float | 0.3 (dashscope) | Sampling temperature |
| `--top-p` | float | — | Top-p sampling |
| `--thinking` | int | 0 | Extended thinking budget |
| `--login` | flag | — | Qwen OAuth login |
| `--logout` | flag | — | Clear credentials |
| `--ndjson` | flag | — | NDJSON bridge mode |
| `--resume` | flag | — | Resume last session |
| `--session-id` | string | — | Resume specific session |
| `--header` | string | — | Extra HTTP header (repeatable) |
| `--verbose` | flag | — | Debug output to stderr |

### 9.2 Environment Variables

| Variable | Description |
|---|---|
| `DASHSCOPE_API_KEY` | DashScope API key |
| `OPENAI_API_KEY` | OpenAI API key (also serves as universal fallback) |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `MODELSCOPE_API_KEY` | ModelScope API key |
| `OPENAI_BASE_URL` | Override API URL |
| `QWEN_MODEL` / `OPENAI_MODEL` | Override default model |

### 9.3 Configuration Resolution Priority

```
CLI arguments > Environment variables > Provider preset defaults
```

### 9.4 Slash Commands

| Command | Description |
|---|---|
| `/exit` `/quit` `/q` | Exit session |
| `/clear` | Clear history, start new session |
| `/model [name]` | Switch/show model |
| `/provider [name]` | Switch provider (auto-updates client) |
| `/thinking [budget]` | Toggle extended thinking |
| `/cost` | Show cumulative token usage and cost estimate |
| `/session` | Show current session info |
| `/login` | Qwen OAuth login |
| `/logout` | Clear OAuth credentials |
| `/help` | Show all commands |
| `/compact` | (Reserved) Conversation summary compression |

---

## 10. Known Limitations

| Limitation | Description | Possible Improvement |
|---|---|---|
| Glob patterns | `fnmatch` does not support `**` recursion | Switch to `pathlib.Path.glob()` |
| Cost estimation | Hardcoded prices, not model-specific | Maintain a model pricing table |
| Token counting | Some providers don't return usage in SSE | Character-based estimation fallback |
| No MCP | Model Context Protocol servers not supported | Implement MCP client |
| No image input | Vision/multimodal input not supported | Add base64 image encoding |
| GIL | Agent parallelism uses threads, subject to GIL | I/O-bound (HTTP requests) are not significantly affected |
| No context compression | No auto-summarization for long conversations | Implement /compact command |
| Parallelism depends on model | Model may return tool_calls one at a time | System prompt guidance, but not guaranteed |
| No readline on Windows | readline module unavailable | Already replaced by SmartInput |

---

## 11. File Structure


Runtime generated:

```
~/.qwen/
└── oauth_creds.json         # Qwen OAuth credential cache

~/.qwen-native/
└── sessions/
    ├── {uuid1}.jsonl         # Session history
    └── {uuid2}.jsonl
```
