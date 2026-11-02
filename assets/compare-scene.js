function mechanismScene(WD) {
  const MCOL0 = ["#0A84FF", "#E8A33D", "#3DAE7C", "#E0637F", "#A66BE0"];
  const MODES = [
    ["bestn", "One call · best of N"],
    ["climber", "One climber"],
    ["swarm", "Swarm"],
    ["sciel", "SCiEL"],
  ];
  const wrap = el("mech3d"),
    cv = el("cmech"),
    ovm = el("ovmech");
  function cols() {
    return (wrap.clientWidth || 1400) < 900 ? 2 : 4;
  }
  const dim = {};
  function size() {
    dim.MW = wrap.clientWidth || 1400;
    const c = cols();
    dim.PW = dim.MW / c;
    dim.PH = Math.round(dim.PW * 0.92);
    dim.MH = dim.PH * (4 / c);
  }
  size();
  const rd = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
  rd.setPixelRatio(Math.min(devicePixelRatio, 2));
  rd.setSize(dim.MW, dim.MH, false);
  rd.setClearColor(0x000000, 1);
  rd.setScissorTest(true);
  const sc = new THREE.Scene();
  sc.fog = new THREE.Fog(0x000000, 140, 300);
  const cam = new THREE.PerspectiveCamera(34, dim.PW / dim.PH, 0.1, 600);
  const V = THREE.Vector3;
  const XL = 84;
  const YL = 18;
  const tg = new V(0, 4, 0);
  const sp = { t: -0.45, p: 0.9, r: (wrap.clientWidth || 1400) < 600 ? 180 : 158 };
  const gl = { t: sp.t, p: sp.p, r: sp.r };
  function plc() {
    cam.position.set(tg.x + sp.r * Math.sin(sp.p) * Math.sin(sp.t), tg.y + sp.r * Math.cos(sp.p), tg.z + sp.r * Math.sin(sp.p) * Math.cos(sp.t));
    cam.lookAt(tg);
  }
  function X(x) {
    return (x * XL) / 2;
  }
  function Z(y) {
    return (-y * XL) / 2;
  }
  function Hh(v) {
    return v * YL;
  }
  function at(p, lift) {
    return new V(X(p[0]), Hh(WD.f(p[0], p[1])) + (lift || 0.35), Z(p[1]));
  }
  ovm.innerHTML = "";
  const panels = MODES.map((m, i) => {
    const d = document.createElement("div");
    d.className = "mpanel";
    d.innerHTML = '<div class="mt">' + m[1] + '</div><div class="mb"></div>';
    ovm.appendChild(d);
    return d;
  });
  function layout() {
    const c = cols();
    panels.forEach((d, i) => {
      d.style.left = (i % c) * dim.PW + "px";
      d.style.top = Math.floor(i / c) * dim.PH + "px";
      d.style.width = dim.PW + "px";
      d.style.height = dim.PH + "px";
    });
  }
  layout();
  addEventListener("resize", () => {
    size();
    rd.setSize(dim.MW, dim.MH, false);
    cam.aspect = dim.PW / dim.PH;
    cam.updateProjectionMatrix();
    layout();
  });
  const MLAB = [];
  function mlab(pi, text, cls, color) {
    const e = document.createElement("div");
    e.className = "l3 " + (cls || "mlab");
    e.textContent = text;
    if (color) e.style.color = color;
    ovm.appendChild(e);
    const L = { pi: pi, el: e, p: new THREE.Vector3(), on: true };
    MLAB.push(L);
    return L;
  }
  const N = 130,
    pos = [],
    col = [],
    ix = [],
    vals = [],
    LOW = new THREE.Color("#0A1828"),
    HIGH = new THREE.Color("#7CC0FF"),
    OFF = new THREE.Color("#121214");
  for (var j = 0; j <= N; j++)
    for (var i = 0; i <= N; i++) {
      const x = -1 + (2 * i) / N,
        y = -1 + (2 * j) / N,
        ok = WD.feasible(x, y),
        v = WD.f(x, y);
      vals.push(ok ? v : -1);
      const c = ok ? LOW.clone().lerp(HIGH, Math.pow(Math.floor(v * 14) / 14, 1.15)) : OFF;
      pos.push(X(x), ok ? Hh(v) : 0, Z(y));
      col.push(c.r, c.g, c.b);
    }
  for (j = 0; j < N; j++)
    for (i = 0; i < N; i++) {
      const a = j * (N + 1) + i;
      ix.push(a, a + 1, a + N + 1, a + 1, a + N + 2, a + N + 1);
    }
  const tg2 = new THREE.BufferGeometry();
  tg2.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  tg2.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  tg2.setIndex(ix);
  sc.add(new THREE.Mesh(tg2, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })));
  function march(levelFn, lv, lift, color, op) {
    const out = [];
    for (j = 0; j < N; j++)
      for (i = 0; i < N; i++) {
        const a0 = j * (N + 1) + i,
          ids = [a0, a0 + 1, a0 + N + 2, a0 + N + 1],
          cs = [
            [i, j],
            [i + 1, j],
            [i + 1, j + 1],
            [i, j + 1],
          ],
          vs = ids.map(levelFn),
          pts = [];
        for (let e = 0; e < 4; e++) {
          const v0 = vs[e],
            v1 = vs[(e + 1) % 4];
          if (v0 < lv !== v1 < lv) {
            const f = (lv - v0) / (v1 - v0),
              p0 = cs[e],
              p1 = cs[(e + 1) % 4];
            pts.push([-1 + (2 * (p0[0] + (p1[0] - p0[0]) * f)) / N, -1 + (2 * (p0[1] + (p1[1] - p0[1]) * f)) / N]);
          }
        }
        for (let q = 0; q + 1 < pts.length; q += 2) {
          const A = pts[q],
            Bq = pts[q + 1];
          out.push(X(A[0]), lift, Z(A[1]), X(Bq[0]), lift, Z(Bq[1]));
        }
      }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(out, 3));
    sc.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: op })));
  }
  [0.2, 0.4, 0.6, 0.8].forEach((lv) => {
    march((k) => vals[k], lv, Hh(lv) + 0.08, 0x9ad0ff, 0.28);
  });
  march((k) => (vals[k] >= 0 ? 1 : 0), 0.5, 0.25, 0x64d2ff, 0.95);
  const glowT = glowTexture();
  const discT = discTexture();
  const ringT = ringTexture();
  const sm = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowT, transparent: true, opacity: 0.75, depthWrite: false }));
  sm.position.copy(at(WD.summit, 0.8));
  sm.scale.set(11, 11, 1);
  sc.add(sm);
  const sd = new THREE.Sprite(new THREE.SpriteMaterial({ map: discT, color: 0xf5f5f7, transparent: true, depthWrite: false }));
  sd.position.copy(at(WD.seed, 0.6));
  sd.scale.set(2.2, 2.2, 1);
  sc.add(sd);
  [0, 1, 2, 3].forEach((pi) => {
    mlab(pi, "seed", "mlab", "#C7C7CC").p.copy(at(WD.seed, 3));
    mlab(pi, "best known", "mlab", "#F5F5F7").p.copy(at(WD.summit, 5));
  });
  mlab(0, "feasible set F", "mlab land", "#64D2FF").p.copy(at([-0.9, 0.62], 1.5));
  mlab(0, "constraint: no solutions here", "mlab land", "#64D2FF").p.copy(new V(X(0.08), 1.6, Z(0.1)));
  mlab(0, "local bump: a trap", "mlab land", "#9AD0FF").p.copy(at([0.02, -0.4], 2.4));
  mlab(0, "narrow ridge to the summit", "mlab land", "#9AD0FF").p.copy(at([0.6, 0.52], 2.6));
  const DL = {
    bestn: { best: mlab(0, "best sample", "mlab", "#F5F5F7") },
    climber: { head: mlab(1, "climber", "mlab", "#F5F5F7") },
    swarm: { heads: [0, 1, 2, 3, 4].map((i) => mlab(2, "climber " + (i + 1), "mlab sm", MCOL0[i])) },
    sciel: {
      heads: [0, 1, 2, 3, 4].map((i) => mlab(3, "deme " + (i + 1), "mlab sm", MCOL0[i])),
      chord: mlab(3, "culture: moved to the leader", "mlab", "#E0637F"),
      screen: mlab(3, "screened before evaluation", "mlab opt", "#98989D"),
      jump: mlab(3, "research probe", "mlab", "#C9A6F0"),
      policy: mlab(3, "policy: where to propose next", "mlab opt", "#C7C7CC"),
    },
  };
  return { wrap, cv, ovm, rd, sc, cam, V, XL, sp, gl, dim, cols, plc, at, panels, MLAB, discT, ringT, DL, MODES, MCOL0 };
}
