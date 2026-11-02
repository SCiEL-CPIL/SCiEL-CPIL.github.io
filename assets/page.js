const VISM = new WeakMap(),
  VIO =
    "IntersectionObserver" in window
      ? new IntersectionObserver(
          (es) => {
            es.forEach((e) => {
              VISM.set(e.target, e.isIntersecting);
            });
          },
          { rootMargin: "200px" },
        )
      : null;
function VIS(n) {
  if (!VIO) return true;
  if (!VISM.has(n)) {
    VISM.set(n, false);
    VIO.observe(n);
  }
  return VISM.get(n) && !document.hidden;
}
const D = DATA;
const R = D.runs;
const NAMES = D.order;
let cur = NAMES[0];
const ISC = ["#E8A33D", "#3DAE7C", "#0A84FF", "#A66BE0", "#2BB3C0", "#D67AB1", "#D1B23F", "#4F7BE8"];
const OPN = { diff: "diff", full_rewrite: "rewrite", cross: "cross", idea: "idea", error_repair: "repair", fix: "repair", seed: "seed" };
const OPC = { diff: "#0A84FF", full_rewrite: "#E8A33D", cross: "#3DAE7C", idea: "#A66BE0", seed: "#8E8E93" };
const RUNC = { "4188325b": "#0A84FF", bfd1c068: "#64D2FF", "8e06385e": "#4F7BE8", aa8fed08: "#E8A33D", dc9726aa: "#D1B23F", "560bd98d": "#E0637F" };
function ic(i) {
  return ISC[(i || 0) % ISC.length];
}
function esc(s) {
  return String(s == null ? "" : s).replace(
    /[&<>"]/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
      })[c],
  );
}
function f4(x) {
  return x == null ? "—" : (+x).toFixed(4);
}
function el(id) {
  return document.getElementById(id);
}
const RUNIDS = {},
  XPU = "https://research-arena.small-pine-3af7.workers.dev/experiments";
function label(n) {
  return "run " + (RUNIDS[n] || n);
}
function LBL(n) {
  return '<a class="xp" href="' + XPU + '" target="_blank" rel="noopener">' + label(n) + "</a>";
}
function XPL(t) {
  return '<a class="xp" href="' + XPU + '" target="_blank" rel="noopener">' + t + "</a>";
}
function dig(s) {
  return -Math.log10(Math.max(1 - s, 1e-5));
}
function byId(r) {
  if (!r._by) {
    r._by = {};
    r.programs.forEach((p) => {
      r._by[p[0]] = p;
    });
  }
  return r._by;
}
function sup(k) {
  return k ? "10⁻" + k : "1";
}
function axisY(s, L, W, Rr, Y, lo, hi) {
  for (let g = Math.ceil(lo); g <= hi; g++) {
    s.push(
      '<line x1="' +
        L +
        '" x2="' +
        (W - Rr) +
        '" y1="' +
        Y(g) +
        '" y2="' +
        Y(g) +
        '" stroke="#2C2C2E"/><text x="' +
        (L - 10) +
        '" y="' +
        (Y(g) + 4) +
        '" fill="#98989D" font-size="11" text-anchor="end">' +
        (g >= 5 ? "best known" : (1 - Math.pow(10, -g)).toFixed(Math.max(1, g))) +
        "</text>",
    );
  }
}

const REPO = "git clone https://github.com/Doldrums/sciel && cd sciel";
const START = {
  api: {
    steps: [
      ["Install the release wheel", "pip install sciel-0.19.0-py3-none-any.whl"],
      ["Set a model key", "export ANTHROPIC_API_KEY=…   # or OPENROUTER_API_KEY"],
      ["Start, then pick the demo, your program or a run to continue", "sciel"],
      ["Or improve one file with solve() and score()", "sciel solve.py --budget 40"],
      ["Or turn a folder of papers, notebooks or data into a checked task", 'sciel task ./materials --note "what to improve"'],
    ],
    note: "With no arguments sciel finds the key, takes the default profile and asks what to improve. Python 3.11–3.14 on Linux or macOS.",
  },
  local: {
    steps: [
      ["Install from the code", REPO + ' && pip install -e ".[tasks]"'],
      ["Serve any open model behind an OpenAI-compatible API", "vllm serve <model> --port 8000"],
      [
        "Point a run at it",
        "cat > local.yaml <<EOF\nextends: configs/circle.yaml\nllm: {api_base: http://127.0.0.1:8000/v1, api_key_env: LOCAL_LLM_KEY, models: [<model>]}\nEOF\nexport LOCAL_LLM_KEY=local",
      ],
      ["Run and watch", "sciel run local.yaml --tasks-root ./tasks"],
    ],
    note: "Our runs use Qwen3.8-27B on two H100s this way. A local model costs electricity rather than tokens.",
  },
};
function tset(k) {
  const d = START[k];
  el("tsteps").innerHTML = d.steps
    .map(
      (st, i) =>
        "<li><span>" +
        esc(st[0]) +
        '</span><div class="cmd"><pre>' +
        esc(st[1]) +
        '</pre><button class="cp" data-c="' +
        esc(st[1]) +
        '">Copy</button></div></li>',
    )
    .join("");
  el("tnote").textContent = d.note;
  [].forEach.call(el("tmode").children, (b) => {
    b.classList.toggle("on", b.dataset.k === k);
  });
}
el("tmode").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) tset(b.dataset.k);
});
el("tsteps").addEventListener("click", (e) => {
  const b = e.target.closest("button.cp");
  if (!b) return;
  try {
    navigator.clipboard.writeText(b.dataset.c).then(() => {
      b.textContent = "Copied";
      setTimeout(() => {
        b.textContent = "Copy";
      }, 1400);
    });
  } catch (x) {}
});
el("startb").addEventListener("click", function () {
  const open = el("startp").hidden;
  el("startp").hidden = !open;
  this.setAttribute("aria-expanded", open);
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".start")) {
    el("startp").hidden = true;
    el("startb").setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    el("startp").hidden = true;
    el("startb").setAttribute("aria-expanded", "false");
  }
});
tset("api");
el("openStart").addEventListener("click", (e) => {
  e.stopPropagation();
  scrollTo({ top: 0, behavior: "smooth" });
  setTimeout(() => {
    el("startp").hidden = false;
    el("startb").setAttribute("aria-expanded", "true");
  }, 450);
});

el("mtoc").addEventListener("click", function (e) {
  e.stopPropagation();
  const o = el("toc").classList.toggle("open");
  this.setAttribute("aria-expanded", o);
});
el("toc").addEventListener("click", (e) => {
  if (e.target.closest("a")) {
    el("toc").classList.remove("open");
    el("mtoc").setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("nav.top")) {
    el("toc").classList.remove("open");
    el("mtoc").setAttribute("aria-expanded", "false");
  }
});
const nav = el("nav"),
  tocA = [].slice.call(document.querySelectorAll("#toc a")),
  secs = tocA.map((a) => document.querySelector(a.getAttribute("href")));
function onScroll() {
  nav.classList.toggle("scrolled", scrollY > 8);
  const y = scrollY + 230;
  let on = -1;
  secs.forEach((s, k) => {
    if (s && s.getBoundingClientRect().top + scrollY <= y) on = k;
  });
  tocA.forEach((a, k) => {
    const was = a.classList.contains("on");
    a.classList.toggle("on", k === on);
    if (k === on && !was) a.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
}
addEventListener("scroll", onScroll, { passive: true });
onScroll();

const listeners = [];
el("runs").innerHTML = NAMES.map((n) => '<button data-r="' + n + '"' + (n === cur ? ' class="on"' : "") + ">" + label(n) + "</button>").join("");
el("runs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  cur = b.dataset.r;
  [].forEach.call(el("runs").children, (x) => {
    x.classList.toggle("on", x === b);
  });
  listeners.forEach((f) => {
    f();
  });
});
const RUNID = {};
function meta() {
  const r = R[cur],
    st = r.engine.steps,
    last = st[st.length - 1],
    isl = {};
  r.programs.forEach((p) => {
    isl[p[2]] = 1;
  });
  el("runmeta").innerHTML =
    '<span class="rl">Figures below</span> run <a class="runlink" href="https://research-arena.small-pine-3af7.workers.dev/experiments" target="_blank" rel="noopener"><code>' +
    cur +
    "</code> ↗</a> · " +
    (r.task === "circles" ? "circle packing" : "third autocorrelation constant") +
    " · " +
    r.programs.length +
    " programs in " +
    Object.keys(isl).length +
    " demes · $" +
    last[1].toFixed(2) +
    " · " +
    (r.hours || "") +
    " h";
}
listeners.push(meta);
meta();

function tipFor(tip, p) {
  if (!p) {
    tip.hidden = true;
    return;
  }
  tip.innerHTML =
    '<div class="t-n">' +
    (p[4] != null ? f4(p[4]) : "no score") +
    '</div><div class="t-m">' +
    p[0] +
    (p[1] ? " ← " + p[1] : " · seed") +
    " · deme " +
    p[2] +
    " · generation " +
    p[3] +
    (p[7] ? " · " + (OPN[p[7]] || p[7]) : "") +
    (p[6] != null ? " · attempt " + p[6] : "") +
    "</div>" +
    (p[8] ? '<div class="t-i">' + esc(p[8]) + "</div>" : "");
  tip.hidden = false;
}
function placeTip(box, tip, e) {
  const b = box.getBoundingClientRect();
  let x = e.clientX - b.left + 16;
  let y = e.clientY - b.top + 16;
  if (x + 340 > b.width) x = Math.max(0, x - 360);
  if (y + 150 > b.height) y = Math.max(0, y - 170);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
function chordLabel(fig) {
  const pth = fig.querySelector("path[data-chord]");
  if (!pth) return;
  const svg = pth.ownerSVGElement;
  const L = pth.getTotalLength();
  const q = pth.getPointAtLength(L * 0.5);
  const q2 = pth.getPointAtLength(L * 0.52);
  let ang = (Math.atan2(q2.y - q.y, q2.x - q.x) * 180) / Math.PI;
  if (ang > 90) ang -= 180;
  if (ang < -90) ang += 180;
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("x", q.x);
  t.setAttribute("y", q.y - 8);
  t.setAttribute("fill", "#E0637F");
  t.setAttribute("font-size", "13");
  t.setAttribute("text-anchor", "middle");
  t.setAttribute("transform", "rotate(" + ang.toFixed(1) + " " + q.x.toFixed(1) + " " + q.y.toFixed(1) + ")");
  t.textContent = "finding passed on";
  svg.appendChild(t);
}
function hover(fig, r) {
  let tip = fig.querySelector(".tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tip";
    tip.hidden = true;
    fig.appendChild(tip);
  }
  const by = byId(r);
  fig.onmousemove = (e) => {
    const id = e.target.getAttribute && e.target.getAttribute("data-id");
    if (!id) {
      tip.hidden = true;
      return;
    }
    tipFor(tip, by[id]);
    placeTip(fig, tip, e);
  };
  fig.onmouseleave = () => {
    tip.hidden = true;
  };
}
