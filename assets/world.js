const G = { amp: 0.14, sig: 0.09, hole: 0.15, np: 4 };
function makeWorld() {
  const PK = [
    [-0.4, -0.28, 0.12],
    [0.02, -0.4, 0.12],
    [-0.45, 0.2, 0.13],
    [0.3, 0.05, 0.12],
    [-0.02, 0.28, 0.12],
    [0.48, 0.36, 0.1],
    [-0.2, -0.05, 0.11],
  ];
  function raw(x, y) {
    const dx = x - 0.72;
    const dy = y - 0.7;
    const d2 = dx * dx + dy * dy;
    let v = 0.78 * Math.exp(-d2 / (1.05 * 1.05)) - 0.08;
    for (let i = 0; i < PK.length; i++) {
      const p = PK[i],
        e = ((x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1])) / (p[2] * p[2]);
      v += G.amp * Math.exp(-e);
    }
    v = Math.max(v, 1.0 * Math.exp(-d2 / (0.07 * 0.07)));
    const ax = 0.5,
      ay = 0.36,
      ux = 0.72 - ax,
      uy = 0.7 - ay,
      L2 = ux * ux + uy * uy,
      t = Math.max(0, Math.min(1, ((x - ax) * ux + (y - ay) * uy) / L2)),
      px = ax + t * ux,
      py = ay + t * uy,
      dd = (x - px) * (x - px) + (y - py) * (y - py);
    v = Math.max(v, (0.7 + 0.27 * t) * Math.exp(-dd / (0.035 * 0.035)));
    return Math.max(0, v);
  }
  function feasible(x, y) {
    if (Math.pow(Math.abs(x) / 0.96, 4) + Math.pow(Math.abs(y) / 0.92, 4) > 1) return false;
    const hx = x - 0.08,
      hy = y - 0.1;
    if (hx * hx + hy * hy < G.hole * G.hole) return false;
    return true;
  }
  function f(x, y) {
    return feasible(x, y) ? raw(x, y) : 0;
  }
  return { f: f, raw: raw, feasible: feasible, seed: [-0.72, -0.62], summit: [0.72, 0.7] };
}
function mkRng(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  const u = Math.max(1e-9, r()),
    v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function runMode(W, mode, B, seed, opt) {
  opt = opt || {};
  const r = mkRng(seed);
  const ev = [];
  let n = 0;
  let best = -1;
  const f = W.f;
  function evalAt(p, isl, from, kind) {
    const v = f(p[0], p[1]);
    n++;
    ev.push({ t: n, k: kind || "eval", i: isl, p: p, from: from, v: v });
    if (v > best) best = v;
    return v;
  }
  const S = W.seed,
    sig = G.sig;
  if (mode === "bestn") {
    for (let q = 0; q < B; q++) {
      var p = [S[0] + gauss(r) * 0.22, S[1] + gauss(r) * 0.22];
      var v = evalAt(p, 0, S, "eval");
      ev[ev.length - 1].acc = false;
    }
    let bi = 0;
    ev.forEach((e, i) => {
      if (e.v > ev[bi].v) bi = i;
    });
    ev[bi].acc = true;
    return { ev: ev, best: best };
  }
  const m = mode === "climber" ? 1 : 5,
    L = [];
  for (var i = 0; i < m; i++) {
    const st = m === 1 ? S.slice() : [S[0] + gauss(r) * 0.12, S[1] + gauss(r) * 0.12];
    L.push({ cur: st, val: f(st[0], st[1]), mu: [0, 0], s: sig, stall: 0, best: st });
  }
  const sci = mode === "sciel",
    useP = sci && opt.policy !== false,
    useW = sci && opt.world !== false,
    useC = sci && opt.culture !== false,
    useR = sci && opt.research !== false;
  const hist = [];
  let turn = 0,
    lastShare = 0;
  function predict(p, c) {
    c = c || p;
    const near = [];
    for (let j = Math.max(0, hist.length - 200); j < hist.length; j++) {
      const h = hist[j],
        d = (h[0] - c[0]) * (h[0] - c[0]) + (h[1] - c[1]) * (h[1] - c[1]);
      if (d < 0.04 && h[2] > 0) near.push(h);
    }
    if (near.length < 5) return 0;
    let n0 = 0,
      sx = 0,
      sy = 0,
      sz = 0,
      sxx = 0,
      syy = 0,
      sxy = 0,
      sxz = 0,
      syz = 0;
    near.forEach((h) => {
      n0++;
      sx += h[0];
      sy += h[1];
      sz += h[2];
      sxx += h[0] * h[0];
      syy += h[1] * h[1];
      sxy += h[0] * h[1];
      sxz += h[0] * h[2];
      syz += h[1] * h[2];
    });
    const mx = sx / n0,
      my = sy / n0,
      mz = sz / n0,
      cxx = sxx / n0 - mx * mx + 1e-4,
      cyy = syy / n0 - my * my + 1e-4,
      cxy = sxy / n0 - mx * my,
      cxz = sxz / n0 - mx * mz,
      cyz = syz / n0 - my * mz,
      det = cxx * cyy - cxy * cxy,
      bx = (cxz * cyy - cyz * cxy) / det,
      by = (cyz * cxx - cxz * cxy) / det;
    return mz + bx * (p[0] - mx) + by * (p[1] - my);
  }
  while (n < B) {
    var i = turn % m;
    const l = L[i];
    turn++;
    if (useC && n - lastShare >= 10 && m > 1) {
      lastShare = n;
      let bj = 0;
      L.forEach((o, j) => {
        if (o.val > L[bj].val) bj = j;
      });
      let worst = 0;
      L.forEach((o, j) => {
        if (o.val < L[worst].val) worst = j;
      });
      if (worst !== bj && L[bj].val - L[worst].val > 0.04) {
        const a = L[worst];
        const b = L[bj];
        var p = [
          a.cur[0] + (b.cur[0] - a.cur[0]) * (0.75 + 0.2 * r()) + gauss(r) * 0.02,
          a.cur[1] + (b.cur[1] - a.cur[1]) * (0.75 + 0.2 * r()) + gauss(r) * 0.02,
        ];
        var v = evalAt(p, worst, b.cur, "chord");
        hist.push([p[0], p[1], v]);
        if (v > a.val) {
          a.cur = p;
          a.val = v;
          a.mu = b.mu.slice();
          a.s = b.s;
          a.stall = 0;
          ev[ev.length - 1].acc = true;
        } else ev[ev.length - 1].acc = false;
        if (n >= B) break;
        continue;
      }
    }
    if (useR && l.stall >= 14) {
      l.stall = 0;
      const dir = [W.summit[0] - l.cur[0], W.summit[1] - l.cur[1]];
      const g = [l.mu[0] || gauss(r) * 0.05, l.mu[1] || gauss(r) * 0.05],
        gn = Math.hypot(g[0], g[1]) || 1;
      let bestJ = null,
        bv = -1;
      for (let k2 = 0; k2 < 3; k2++) {
        const jl = 0.25 + 0.25 * r(),
          ang = Math.atan2(g[1], g[0]) + (r() - 0.5) * 1.6,
          pj = [l.cur[0] + Math.cos(ang) * jl, l.cur[1] + Math.sin(ang) * jl];
        let pv = useW ? predict(pj, l.cur) : r();
        if (!(pv > -1e9)) pv = -1;
        if (!bestJ || pv > bv) {
          bv = pv;
          bestJ = pj;
        }
      }
      var v = evalAt(bestJ, i, l.cur, "jump");
      hist.push([bestJ[0], bestJ[1], v]);
      if (v > l.val) {
        l.cur = bestJ;
        l.val = v;
        ev[ev.length - 1].acc = true;
      } else ev[ev.length - 1].acc = false;
      continue;
    }
    const props = [],
      np = useW ? G.np : 1;
    for (let k = 0; k < np; k++) {
      const s = useP ? l.s : sig;
      var p = [l.cur[0] + (useP ? l.mu[0] : 0) + gauss(r) * s, l.cur[1] + (useP ? l.mu[1] : 0) + gauss(r) * s];
      props.push(p);
    }
    let pick = props[0];
    if (useW) {
      let bp = -1;
      props.forEach((p) => {
        let v = predict(p, l.cur);
        if (!(v > -1e9)) v = -1;
        if (v > bp) {
          bp = v;
          pick = p;
        }
      });
      props.forEach((p) => {
        if (p !== pick) ev.push({ t: n, k: "screen", i: i, p: p, from: l.cur, v: null });
      });
    }
    const from = l.cur;
    var v = evalAt(pick, i, from, "eval");
    hist.push([pick[0], pick[1], v]);
    if (v > l.val) {
      if (useP) {
        l.mu = [0.6 * l.mu[0] + 0.4 * (pick[0] - from[0]), 0.6 * l.mu[1] + 0.4 * (pick[1] - from[1])];
        l.s = Math.min(0.12, l.s * 1.1);
      }
      l.cur = pick;
      l.val = v;
      l.stall = 0;
      ev[ev.length - 1].acc = true;
    } else {
      if (useP) {
        l.s = Math.max(0.025, l.s * 0.96);
        l.mu = [l.mu[0] * 0.9, l.mu[1] * 0.9];
      }
      l.stall++;
      ev[ev.length - 1].acc = false;
    }
    ev[ev.length - 1].pol = { i: i, c: l.cur.slice(), s: useP ? l.s : sig, mu: useP ? l.mu.slice() : [0, 0] };
  }
  return { ev: ev, best: best, L: L };
}
if (typeof module !== "undefined") module.exports = { makeWorld: makeWorld, runMode: runMode, G: G };

function canvasTexture(draw) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  draw(c.getContext("2d"));
  return new THREE.CanvasTexture(c);
}
function glowTexture() {
  return canvasTexture((g) => {
    const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    r.addColorStop(0, "rgba(255,255,255,1)");
    r.addColorStop(0.3, "rgba(255,255,255,.35)");
    r.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = r;
    g.fillRect(0, 0, 64, 64);
  });
}
function discTexture() {
  return canvasTexture((g) => {
    g.beginPath();
    g.arc(32, 32, 24, 0, 7);
    g.fillStyle = "#fff";
    g.fill();
    g.lineWidth = 5;
    g.strokeStyle = "#000";
    g.stroke();
  });
}
function ringTexture() {
  return canvasTexture((g) => {
    g.beginPath();
    g.arc(32, 32, 20, 0, 7);
    g.lineWidth = 7;
    g.strokeStyle = "#fff";
    g.stroke();
  });
}
