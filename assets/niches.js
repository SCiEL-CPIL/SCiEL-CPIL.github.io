(() => {
  const V = D.vq,
    fin = D.niches.final,
    K = V.xy.length,
    occ = [];
  for (let k = 0; k < K; k++) occ.push({ isl: {}, best: 0 });
  fin.forEach((i) => {
    i.grid.forEach((c) => {
      const o = occ[c[0]];
      if (!o) return;
      o.isl[i.idx] = 1;
      o.best = Math.max(o.best, c[2]);
    });
  });
  const xs = V.xy.map((p) => p[0]),
    ys = V.xy.map((p) => p[1]),
    x0 = Math.min.apply(null, xs),
    x1 = Math.max.apply(null, xs),
    y0 = Math.min.apply(null, ys),
    y1 = Math.max.apply(null, ys);
  const W = 520,
    H = 470,
    P = 78;
  function X(v) {
    return P + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * P);
  }
  function Y(v) {
    return P - 10 + (1 - (v - y0) / (y1 - y0 || 1)) * (H - 2 * P);
  }
  const pos = V.xy.map((p) => [X(p[0]), Y(p[1])]),
    home = pos.map((p) => p.slice()),
    rad = occ.map((o) => 7 + Object.keys(o.isl).length * 2.6);
  for (let it = 0; it < 400; it++) {
    for (let a = 0; a < K; a++)
      for (let b = a + 1; b < K; b++) {
        const dx = pos[b][0] - pos[a][0],
          dy = pos[b][1] - pos[a][1],
          d = Math.hypot(dx, dy) || 0.01,
          need = rad[a] + rad[b] + 26;
        if (d < need) {
          const m = (need - d) / 2,
            ux = dx / d,
            uy = dy / d;
          pos[a][0] -= ux * m;
          pos[a][1] -= uy * m;
          pos[b][0] += ux * m;
          pos[b][1] += uy * m;
        }
      }
    for (let c = 0; c < K; c++) {
      pos[c][0] += (home[c][0] - pos[c][0]) * 0.02;
      pos[c][1] += (home[c][1] - pos[c][1]) * 0.02;
      pos[c][0] = Math.max(40, Math.min(W - 40, pos[c][0]));
      pos[c][1] = Math.max(40, Math.min(H - 60, pos[c][1]));
    }
  }
  function PX(k) {
    return pos[k][0];
  }
  function PY(k) {
    return pos[k][1];
  }
  const s = ['<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Twelve learned niches">'],
    L0 = 40,
    B0 = 40;
  s.push('<rect x="' + L0 + '" y="12" width="' + (W - L0 - 12) + '" height="' + (H - 12 - B0) + '" fill="#0B0B0C" stroke="#2C2C2E"/>');
  s.push(
    '<text x="' +
      (L0 + (W - L0) / 2) +
      '" y="' +
      (H - 12) +
      '" fill="#98989D" font-size="12" text-anchor="middle">codebook, component 1</text><text x="16" y="' +
      ((H - B0) / 2 + 6) +
      '" fill="#98989D" font-size="12" text-anchor="middle" transform="rotate(-90 16 ' +
      ((H - B0) / 2 + 6) +
      ')">codebook, component 2</text>',
  );
  V.xy.forEach((p, i) => {
    const d = V.xy
      .map((q, j) => [j, Math.hypot(q[0] - p[0], q[1] - p[1])])
      .filter((a) => a[0] !== i)
      .sort((a, b) => a[1] - b[1]);
    d.slice(0, 2).forEach((a) => {
      if (a[0] > i) s.push('<line x1="' + PX(i) + '" y1="' + PY(i) + '" x2="' + PX(a[0]) + '" y2="' + PY(a[0]) + '" stroke="#2C2C2E"/>');
    });
  });
  occ.forEach((o, k) => {
    const isl = Object.keys(o.isl).map(Number),
      n = isl.length,
      cx = PX(k),
      cy = PY(k),
      r = rad[k];
    s.push(
      '<circle cx="' +
        cx +
        '" cy="' +
        cy +
        '" r="' +
        r +
        '" fill="#000" stroke="' +
        (n ? "#48484D" : "#2C2C2E") +
        '"' +
        (n ? "" : ' stroke-dasharray="2 3"') +
        "/>",
    );
    isl.forEach((ii, t) => {
      const a = -Math.PI / 2 + (t / Math.max(1, n)) * Math.PI * 2,
        rr = n > 1 ? r * 0.55 : 0;
      s.push(
        '<circle cx="' +
          (cx + Math.cos(a) * rr).toFixed(1) +
          '" cy="' +
          (cy + Math.sin(a) * rr).toFixed(1) +
          '" r="3.4" fill="' +
          ic(ii) +
          '" stroke="#000" stroke-width=".8"/>',
      );
    });
    s.push('<text x="' + cx + '" y="' + (cy + r + 13) + '" fill="#98989D" font-size="10.5" text-anchor="middle">N' + k + "</text>");
    s.push(
      '<circle cx="' +
        cx +
        '" cy="' +
        cy +
        '" r="' +
        (r + 4) +
        '" fill="transparent"><title>niche ' +
        k +
        ": " +
        (n ? n + " demes hold an elite here; best " + o.best.toFixed(4) : "empty at the end of the run") +
        "</title></circle>",
    );
  });
  s.push("</svg>");
  el("vqmap").insertAdjacentHTML("beforeend", s.join(""));
})();

(() => {
  const NI = D.niches;
  const C = NI.counts || {};
  const fin = NI.final;
  let cells = 0;
  fin.forEach((i) => {
    cells += i.grid.length;
  });
  function cn(k) {
    return C[k] || 0;
  }
  el("s-niche").innerHTML =
    "In " +
    XPL("this run") +
    " " +
    fin.length +
    " demes held " +
    cells +
    " occupied cells at the end, across twelve learned niches; the codebook was re-fitted " +
    cn("niches/rebuild") +
    " times as the population moved.";
  const M = [
    [
      "Niches per deme",
      fin.length + " demes · " + cells + " cells",
      'MAP-Elites over learned niches: a VQ codebook of up to twelve centroids in an embedding of the code, crossed with score bins. <span class="key">One elite per cell, so a new idea competes only with its own niche.</span>',
    ],
    [
      "Demes spawned mid-run",
      cn("dynamic_islands/spawn") + "",
      'Aimed at unexplored regions, chosen by a diffusion target, by density or by an untried paradigm. <span class="key">A new deme starts where the others are not.</span>',
    ],
    [
      "Novelty, run-wide",
      cn("candidate_novelty/accept") + " accepted · " + cn("candidate_novelty/reject") + " rejected",
      'Repeats are caught by hash, AST and output before they cost an evaluation, <span class="key">so the budget goes to new guesses</span>.',
    ],
    [
      "Smoke → full → confirm",
      (NI.reached.full || 0) + " reached full",
      "A cheap smoke test first, the full stage next, then re-measurement of records: " +
        cn("confirmation/consistent") +
        " agreed, " +
        cn("confirmation/inconsistent") +
        ' did not, and <span class="key">the worst repeat is kept</span>.',
    ],
    [
      "Parallel waves",
      cn("parallel_search/wave_dispatch") + " waves · " + cn("parallel_search/job_committed") + " jobs",
      'Demes propose in parallel against shared capacity; every job is journaled, <span class="key">so a crashed run replays and continues</span>.',
    ],
    [
      "Population view",
      " 1.55 → 1.462",
      'A candidate imports its deme’s best construction and the task’s own objective while it runs, <span class="key">so it continues the measured object instead of rebuilding it</span>. On this task: the bound after the same 50 evaluations.',
    ],
  ];
  el("mach").innerHTML = M.map((m) => "<div><b>" + m[0] + '</b><div class="n">' + m[1] + "</div><p>" + m[2] + "</p></div>").join("");
  if (!window.THREE) return;
  const wrap = el("niche3d"),
    cv = el("cniche"),
    ov = el("ovniche"),
    tip = el("tipniche");
  let W = wrap.clientWidth || 1000;
  const MOB = W < 560;
  let H = MOB ? Math.round(W * 1.2) : Math.round(Math.max(360, Math.min(560, W * 0.62)));
  const rd = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
  rd.setPixelRatio(Math.min(devicePixelRatio, 2));
  rd.setSize(W, H, false);
  rd.setClearColor(0x000000, 1);
  const sc = new THREE.Scene();
  sc.fog = new THREE.Fog(0x000000, 120, 260);
  const cam = new THREE.PerspectiveCamera(30, W / H, 0.1, 600);
  const V = THREE.Vector3;
  const tg = new V(0, 2, MOB ? 9 : 2);
  const sp = { t: MOB ? -0.04 : -0.12, p: MOB ? 0.62 : 0.9, r: MOB ? 150 : 142 };
  const gl = { t: sp.t, p: sp.p, r: sp.r };
  let dr = null;
  let touched = false;
  function plc() {
    cam.position.set(tg.x + sp.r * Math.sin(sp.p) * Math.sin(sp.t), tg.y + sp.r * Math.cos(sp.p), tg.z + sp.r * Math.sin(sp.p) * Math.cos(sp.t));
    cam.lookAt(tg);
  }
  addEventListener("resize", () => {
    W = wrap.clientWidth || W;
    H = MOB ? Math.round(W * 1.2) : Math.round(Math.max(360, Math.min(560, W * 0.62)));
    rd.setSize(W, H, false);
    cam.aspect = W / H;
    cam.updateProjectionMatrix();
  });
  const CS = 1.55,
    GX = 12,
    GZ = 10,
    BW = GX * CS,
    BD = GZ * CS,
    GAP = 5,
    COLS = MOB ? 2 : 4,
    ROWS = 8 / COLS;
  function origin(k) {
    const c = k % COLS,
      r = Math.floor(k / COLS);
    return new V((c - (COLS - 1) / 2) * (BW + GAP), 0, (r - (ROWS - 1) / 2) * (BD + GAP + (MOB ? 3 : 0)));
  }
  const labs = [];
  function lab(t, p, color) {
    const e = document.createElement("div");
    e.className = "l3 mlab";
    e.textContent = t;
    if (color) e.style.color = color;
    ov.appendChild(e);
    const L = { el: e, p: p };
    labs.push(L);
    return L;
  }
  const base = new THREE.Group();
  sc.add(base);
  for (let k = 0; k < 8; k++) {
    const o = origin(k),
      g = new THREE.BufferGeometry(),
      pts = [];
    for (let i = 0; i <= GX; i++) pts.push(o.x - BW / 2 + i * CS, 0, o.z - BD / 2, o.x - BW / 2 + i * CS, 0, o.z + BD / 2);
    for (let j = 0; j <= GZ; j++) pts.push(o.x - BW / 2, 0, o.z - BD / 2 + j * CS, o.x + BW / 2, 0, o.z - BD / 2 + j * CS);
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    base.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x2a2a2e, transparent: true, opacity: 0.8 })));
  }
  function dig(s) {
    return -Math.log10(Math.max(1 - s, 1e-4));
  }
  let dyn = null,
    bars = [],
    bmeta = [],
    ilabs = [];
  const SN = NI.snaps.concat([[NI.snaps.length ? NI.snaps[NI.snaps.length - 1][0] + 20 : 200, NI.final]]);
  let si = SN.length - 1;
  let playing = true;
  let timer = null;
  function draw(idx) {
    const snap = SN[idx];
    el("nIt").textContent = "attempt " + Math.min(snap[0], 200);
    if (dyn) {
      sc.remove(dyn);
      dyn.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
    }
    dyn = new THREE.Group();
    sc.add(dyn);
    bars = [];
    bmeta = [];
    ilabs.forEach((L) => {
      L.el.remove();
      labs.splice(labs.indexOf(L), 1);
    });
    ilabs = [];
    snap[1].forEach((isl) => {
      const o = origin(isl.idx),
        col = new THREE.Color(ic(isl.idx));
      isl.grid.forEach((c) => {
        const s = c[2],
          h = s > 0 ? Math.max(0.35, (dig(s) / 3) * 13) : 0.12,
          geo = new THREE.BoxGeometry(CS * 0.82, h, CS * 0.82),
          m = new THREE.Mesh(
            geo,
            new THREE.MeshBasicMaterial({ color: s > 0 ? col : new THREE.Color("#48484D"), transparent: true, opacity: 0.55, depthWrite: false }),
          );
        m.position.set(o.x - BW / 2 + (c[0] + 0.5) * CS, h / 2, o.z + BD / 2 - (c[1] + 0.5) * CS);
        dyn.add(m);
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.95 }));
        e.position.copy(m.position);
        dyn.add(e);
        bars.push(m);
        bmeta.push({ isl: isl.idx, c: c });
      });
      ilabs.push(
        lab(
          MOB ? "deme " + isl.idx + " · " + (+isl.best).toFixed(4) : "deme " + isl.idx + " · " + isl.grid.length + " cells · best " + (+isl.best).toFixed(4),
          new V(o.x, 0.2, o.z + BD / 2 + 3.2),
          ic(isl.idx),
        ),
      );
    });
  }
  function step() {
    if (!playing) return;
    si = (si + 1) % SN.length;
    draw(si);
    timer = setTimeout(step, si === SN.length - 1 ? 3200 : 1300);
  }
  draw(si);
  const io = new IntersectionObserver(
    (es) => {
      es.forEach((en) => {
        if (en.isIntersecting && !wrap.dataset.on) {
          wrap.dataset.on = 1;
          si = -1;
          step();
        }
      });
    },
    { threshold: 0.3 },
  );
  io.observe(wrap);
  el("nplay").addEventListener("click", function () {
    playing = !playing;
    this.textContent = playing ? "Pause" : "Play";
    clearTimeout(timer);
    if (playing) step();
  });
  cv.addEventListener("pointerdown", (e) => {
    dr = { x: e.clientX, y: e.clientY, t: gl.t, p: gl.p };
    touched = true;
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointerup", () => {
    dr = null;
  });
  cv.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      gl.r = Math.max(70, Math.min(240, gl.r * (1 + e.deltaY * 0.001)));
    },
    { passive: false },
  );
  const ray = new THREE.Raycaster(),
    mo = new THREE.Vector2();
  cv.addEventListener("pointermove", (e) => {
    if (dr) {
      gl.t = dr.t - (e.clientX - dr.x) * 0.006;
      gl.p = Math.max(0.15, Math.min(1.45, dr.p - (e.clientY - dr.y) * 0.005));
      tip.hidden = true;
      return;
    }
    const b = cv.getBoundingClientRect();
    mo.x = ((e.clientX - b.left) / b.width) * 2 - 1;
    mo.y = (-(e.clientY - b.top) / b.height) * 2 + 1;
    ray.setFromCamera(mo, cam);
    const hit = ray.intersectObjects(bars)[0];
    if (!hit) {
      tip.hidden = true;
      return;
    }
    const m = bmeta[bars.indexOf(hit.object)];
    tip.innerHTML =
      '<div class="t-n">' +
      (m.c[2] > 0 ? (+m.c[2]).toFixed(4) : "scored zero") +
      '</div><div class="t-m">deme ' +
      m.isl +
      " · cell (" +
      m.c[0] +
      ", " +
      m.c[1] +
      ")</div>";
    tip.hidden = false;
    placeTip(wrap, tip, e);
  });
  cv.addEventListener("pointerleave", () => {
    tip.hidden = true;
  });
  const v3 = new V();
  (function fr() {
    if (!VIS(wrap)) {
      requestAnimationFrame(fr);
      return;
    }
    if (!touched) gl.t += 0.0006;
    sp.t += (gl.t - sp.t) * 0.12;
    sp.p += (gl.p - sp.p) * 0.12;
    sp.r += (gl.r - sp.r) * 0.12;
    plc();
    rd.render(sc, cam);
    labs.forEach((L) => {
      v3.copy(L.p).project(cam);
      if (v3.z > 1) {
        L.el.style.transform = "translate(-9999px,-9999px)";
        return;
      }
      L.el.style.transform = "translate(" + (((v3.x + 1) / 2) * W).toFixed(1) + "px," + (((1 - v3.y) / 2) * H).toFixed(1) + "px)";
    });
    requestAnimationFrame(fr);
  })();
})();
