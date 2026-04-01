# 🤖 SmartBlog AI Agent

An end-to-end, multi-agent AI system designed to autonomously research, write, and format high-quality, production-ready blog posts. Built with LangGraph, this system leverages advanced LLMs for reasoning and writing, live web search for factual grounding, and free APIs for dynamic image generation. 

The architecture separates the AI orchestration logic into a robust **FastAPI** backend, which serves generated markdown content to a clean, interactive **Streamlit** frontend.

## ✨ Key Features

* **Multi-Agent Orchestration:** Utilizes LangGraph to break down the writing process into distinct nodes (Router -> Researcher -> Orchestrator -> Writer -> Reducer/Image Generator).
* **Live Factual Grounding:** Integrates Tavily Search API to pull up-to-date information, completely preventing AI hallucinations on recent topics.
* **Intelligent Routing:** Automatically decides if a topic requires "open book" web research (e.g., recent news) or can be written "closed book" (e.g., evergreen tutorials).
* **Free & Fast Text Generation:** Powered by Groq's lightning-fast LLaMA 3 models for high-quality, token-efficient reasoning and writing.
* **Zero-Cost Image Generation:** Uses Pollinations AI to dynamically generate context-aware blog images without requiring complex API keys or hitting restrictive rate limits.
* **Production-Ready Architecture:** Clean separation of concerns with a RESTful FastAPI backend and a Streamlit client, fully containerized using Docker.

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **AI & Orchestration:** LangGraph, LangChain, Groq (Llama 3.3)
* **Web Search Tool:** Tavily API
* **Image Generation:** Pollinations AI (Open Source)
* **Deployment:** Docker, Git

## ⚙️ How It Works

1. **User Input:** A topic is submitted via the Streamlit UI.
2. **Router Node:** Analyzes the prompt and decides if live web research is required.
3. **Research Node (Optional):** Queries the web via Tavily and extracts high-signal evidence.
4. **Orchestrator Node:** Creates a structured, multi-section outline (Plan) with target word counts and specific instructions.
5. **Worker Nodes:** Generates markdown content for each section in parallel, heavily citing the gathered evidence.
6. **Reducer & Image Node:** Merges the sections, decides where visual aids are needed, and fetches relevant images via Pollinations AI.
7. **Delivery:** The final Markdown text (with image links) is sent back to the Streamlit UI for preview and download.

## 🚀 Local Setup & Installation

Follow these steps to run the SmartBlog AI Agent on a local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/vijay-sonavane9/smartblog-ai-agent.git](https://github.com/vijay-sonavane9/smartblog-ai-agent.git)
cd smartblog-ai-agent
