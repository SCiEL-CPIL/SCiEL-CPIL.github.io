if (window.THREE)
  (() => {
    const wrap = el("map3d"),
      canvas = el("c3d"),
      ov = el("ov3d"),
      tip = el("tip3d");
    let W = wrap.clientWidth || 1400,
      H = Math.round(Math.max(420, Math.min(720, W * 0.52)));
    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(W, H, false);
    renderer.setClearColor(0x000000, 1);
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x000000, 170, 330);
    const camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 600);
    const XL = 60,
      ZL = 44,
      YL = 24,
      target = new THREE.Vector3(0, YL * 0.4, 0);
    const VIEWS = {
      corner: { t: -0.62, p: 1.08, r: 112 },
      front: { t: 0.0001, p: 1.38, r: 104 },
      side: { t: -Math.PI / 2, p: 1.3, r: 104 },
      top: { t: -0.0001, p: 0.02, r: 118 },
      low: { t: -0.9, p: 1.5, r: 96 },
      orbit: { t: -0.62, p: 1.0, r: 118 },
    };
    const sph = { t: VIEWS.corner.t, p: VIEWS.corner.p, r: VIEWS.corner.r };
    let goal = { t: sph.t, p: sph.p, r: sph.r };
    let drag = null;
    let touched = false;
    function place() {
      camera.position.set(
        target.x + sph.r * Math.sin(sph.p) * Math.sin(sph.t),
        target.y + sph.r * Math.cos(sph.p),
        target.z + sph.r * Math.sin(sph.p) * Math.cos(sph.t),
      );
      camera.lookAt(target);
    }
    window.addEventListener("resize", () => {
      W = wrap.clientWidth || W;
      H = Math.round(Math.max(340, Math.min(720, W * (W < 760 ? 0.95 : 0.52))));
      renderer.setSize(W, H, false);
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
    });
    const disc = discTexture();
    const glow = glowTexture();
    let group = null,
      labels = [],
      pts = null,
      meta3 = [],
      amode = "disk",
      smode = "surface",
      orbit = false;
    function lab(text, pos, cls) {
      const e = document.createElement("div");
      e.className = "l3 " + (cls || "");
      e.textContent = text;
      ov.appendChild(e);
      labels.push({ el: e, p: pos });
      return e;
    }
    function line(a, color, op, dashed) {
      const geo = new THREE.BufferGeometry().setFromPoints(a);
      const m = dashed
        ? new THREE.LineDashedMaterial({ color: color, transparent: true, opacity: op, dashSize: 0.6, gapSize: 0.9 })
        : new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: op });
      const l = new THREE.Line(geo, m);
      if (dashed) l.computeLineDistances();
      group.add(l);
      return l;
    }
    const LOW = new THREE.Color("#0B1626"),
      HIGH = new THREE.Color("#5AB0FF");
    function coordsFor(r, m) {
      if (m === "code" || m === "landscape") return r.coords[m] || {};
      const out = {};
      if (m === "disk") {
        const dm = diskModel(r);
        Object.keys(dm.dots).forEach((id) => {
          out[id] = [dm.dots[id].x, dm.dots[id].y];
        });
        return out;
      }
      let mx = 1,
        ni = 1;
      r.programs.forEach((p) => {
        mx = Math.max(mx, p[6] || 0);
        ni = Math.max(ni, (p[2] || 0) + 1);
      });
      r.programs.forEach((p) => {
        out[p[0]] = [0.03 + (0.94 * (p[6] || 0)) / mx, 0.06 + (0.88 * ((p[2] || 0) + 0.5)) / ni];
      });
      return out;
    }
    const AXN = {
      code: ["code, component 1", "code, component 2"],
      landscape: null,
      disk: ["culture disk", "sector = deme · radius = generation"],
      time: ["attempt", "deme"],
    };
    function build() {
      if (group) {
        scene.remove(group);
        group.traverse((o) => {
          if (o.geometry) o.geometry.dispose();
          if (o.material) o.material.dispose();
        });
      }
      labels.forEach((L) => {
        L.el.remove();
      });
      labels = [];
      group = new THREE.Group();
      scene.add(group);
      const r = R[cur],
        by = byId(r),
        items = [],
        c = coordsFor(r, amode);
      Object.keys(c).forEach((id) => {
        const p = by[id];
        if (p && p[4] > 0) items.push({ id: id, p: p, x: c[id][0], y: c[id][1], d: dig(p[4]) });
      });
      if (!items.length) return;
      const ds = items.map((i) => i.d).sort((a, b) => a - b);
      let dlo = ds[Math.floor(ds.length * 0.04)];
      let dhi = ds[ds.length - 1];
      dlo = Math.floor(dlo * 2) / 2;
      dhi = Math.ceil(dhi * 2) / 2;
      if (dhi <= dlo) dhi = dlo + 1;
      function X(x) {
        return (x - 0.5) * XL;
      }
      function Z(y) {
        return (0.5 - y) * ZL;
      }
      function Yh(d) {
        return Math.max(0, (d - dlo) / (dhi - dlo)) * YL;
      }
      const N = 56,
        pos = [],
        col = [],
        ok = [],
        ix = [],
        hs = [];
      for (var j = 0; j <= N; j++)
        for (var i = 0; i <= N; i++) {
          const u = i / N;
          const v = j / N;
          let ws = 0;
          let wv = 0;
          let near = 9;
          items.forEach((q) => {
            const dd = Math.hypot(q.x - u, q.y - v);
            near = Math.min(near, dd);
            const w = 1 / (Math.pow(dd, 3) + 1e-5);
            ws += w;
            wv += w * q.d;
          });
          const h = wv / ws,
            t = Math.max(0, Math.min(1, (h - dlo) / (dhi - dlo))),
            band = Math.floor(t * 10) / 10;
          const cc = LOW.clone().lerp(HIGH, Math.pow(band, 1.2));
          pos.push(X(u), Yh(h), Z(v));
          col.push(cc.r, cc.g, cc.b);
          ok.push(amode === "landscape" || near < (amode === "time" ? 0.05 : 0.075));
          hs.push(h);
        }
      for (j = 0; j < N; j++)
        for (i = 0; i < N; i++) {
          const a = j * (N + 1) + i,
            b = a + 1,
            d2 = a + N + 1,
            e2 = d2 + 1;
          if (ok[a] && ok[b] && ok[d2]) ix.push(a, d2, b);
          if (ok[b] && ok[e2] && ok[d2]) ix.push(b, d2, e2);
        }
      const sg = new THREE.BufferGeometry();
      sg.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      sg.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
      sg.setIndex(ix);
      if (smode !== "points")
        group.add(
          new THREE.Mesh(
            sg,
            new THREE.MeshBasicMaterial({
              vertexColors: true,
              transparent: true,
              opacity: smode === "contours" ? 0.12 : 0.62,
              side: THREE.DoubleSide,
              depthWrite: false,
            }),
          ),
        );
      if (smode === "contours") {
        const cl = [],
          cc2 = [];
        for (let lv = Math.ceil(dlo * 4) / 4; lv < dhi; lv += 0.25) {
          const tc = Math.max(0, Math.min(1, (lv - dlo) / (dhi - dlo))),
            ccol = LOW.clone().lerp(new THREE.Color("#9AD0FF"), 0.25 + 0.75 * tc),
            yy = Yh(lv);
          for (j = 0; j < N; j++)
            for (i = 0; i < N; i++) {
              const a0 = j * (N + 1) + i,
                a1 = a0 + 1,
                a3 = a0 + N + 1,
                a2 = a3 + 1;
              if (!(ok[a0] && ok[a1] && ok[a2] && ok[a3])) continue;
              const vs = [hs[a0], hs[a1], hs[a2], hs[a3]],
                cs = [
                  [i, j],
                  [i + 1, j],
                  [i + 1, j + 1],
                  [i, j + 1],
                ],
                ptsC = [];
              for (let e = 0; e < 4; e++) {
                const v0 = vs[e],
                  v1 = vs[(e + 1) % 4];
                if (v0 < lv !== v1 < lv) {
                  const f = (lv - v0) / (v1 - v0),
                    p0 = cs[e],
                    p1 = cs[(e + 1) % 4];
                  ptsC.push([(p0[0] + (p1[0] - p0[0]) * f) / N, (p0[1] + (p1[1] - p0[1]) * f) / N]);
                }
              }
              for (let q2 = 0; q2 + 1 < ptsC.length; q2 += 2) {
                cl.push(X(ptsC[q2][0]), yy, Z(ptsC[q2][1]), X(ptsC[q2 + 1][0]), yy, Z(ptsC[q2 + 1][1]));
                cc2.push(ccol.r, ccol.g, ccol.b, ccol.r, ccol.g, ccol.b);
              }
            }
        }
        const cg = new THREE.BufferGeometry();
        cg.setAttribute("position", new THREE.Float32BufferAttribute(cl, 3));
        cg.setAttribute("color", new THREE.Float32BufferAttribute(cc2, 3));
        group.add(new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })));
      }
      line(
        [
          new THREE.Vector3(X(0), 0, Z(0)),
          new THREE.Vector3(X(1), 0, Z(0)),
          new THREE.Vector3(X(1), 0, Z(1)),
          new THREE.Vector3(X(0), 0, Z(1)),
          new THREE.Vector3(X(0), 0, Z(0)),
        ],
        0x48484d,
        0.8,
      );
      for (let g = 1; g < 5; g++) {
        line([new THREE.Vector3(X(g / 5), 0, Z(0)), new THREE.Vector3(X(g / 5), 0, Z(1))], 0x2a2a2e, 0.5, true);
        line([new THREE.Vector3(X(0), 0, Z(g / 5)), new THREE.Vector3(X(1), 0, Z(g / 5))], 0x2a2a2e, 0.5, true);
      }
      const ax = (amode === "landscape" ? r.axes.landscape : AXN[amode]) || ["", ""];
      lab(ax[0], new THREE.Vector3(0, 0, Z(0) + 5), "title");
      lab(ax[1], new THREE.Vector3(X(1) + 6, 0, 0), "title");
      const hx = X(0),
        hz = Z(1);
      line([new THREE.Vector3(hx, 0, hz), new THREE.Vector3(hx, YL + 1, hz)], 0x48484d, 0.8);
      for (let k = Math.ceil(dlo); k <= dhi; k++) {
        const y = Yh(k);
        line([new THREE.Vector3(hx - 0.7, y, hz), new THREE.Vector3(hx + 0.7, y, hz)], 0x48484d, 0.9);
        line([new THREE.Vector3(X(0), y, hz), new THREE.Vector3(X(1), y, hz)], 0x2a2a2e, 0.35, true);
        lab("1 − score = " + sup(k), new THREE.Vector3(hx - 1.6, y, hz), "lvl");
      }
      const pmap = {};
      items.forEach((q) => {
        pmap[q.id] = new THREE.Vector3(X(q.x), Yh(q.d), Z(q.y));
      });
      const lp = [];
      items.forEach((q) => {
        const pa = q.p[1] && pmap[q.p[1]];
        if (pa) lp.push(pa.x, pa.y, pa.z, pmap[q.id].x, pmap[q.id].y, pmap[q.id].z);
      });
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
      group.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xc7c7cc, transparent: true, opacity: 0.28 })));
      const pp = [],
        pc = [];
      meta3 = [];
      items.forEach((q) => {
        const v3 = pmap[q.id],
          cc = new THREE.Color(ic(q.p[2]));
        pp.push(v3.x, v3.y, v3.z);
        pc.push(cc.r, cc.g, cc.b);
        meta3.push(q.p);
      });
      const pg = new THREE.BufferGeometry();
      pg.setAttribute("position", new THREE.Float32BufferAttribute(pp, 3));
      pg.setAttribute("color", new THREE.Float32BufferAttribute(pc, 3));
      pts = new THREE.Points(pg, new THREE.PointsMaterial({ size: 2.4, map: disc, vertexColors: true, transparent: true, alphaTest: 0.3, depthWrite: false }));
      group.add(pts);
      const chainIds = r.chain.map((c) => c.child).filter((id) => pmap[id]);
      if (chainIds.length > 1)
        line(
          chainIds.map((id) => pmap[id]),
          0xf5f5f7,
          0.85,
        );
      chainIds.forEach((id) => {
        const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, transparent: true, opacity: 0.35, depthWrite: false }));
        s.position.copy(pmap[id]);
        s.scale.set(3, 3, 1);
        group.add(s);
      });
      if (chainIds.length) lab(f4(by[chainIds[chainIds.length - 1]][4]), pmap[chainIds[chainIds.length - 1]], "rec");
      const isl = {};
      items.forEach((q) => {
        isl[q.p[2]] = 1;
      });
      el("leg3d").innerHTML =
        Object.keys(isl)
          .sort((a, b) => a - b)
          .map((k) => '<span><i style="background:' + ic(+k) + '"></i>deme ' + k + "</span>")
          .join("") +
        '<span><i class="ln" style="background:#C7C7CC"></i>edit, parent to child</span><span><i class="ln" style="background:#F5F5F7"></i>records</span>' +
        (amode === "code" && r.codecap ? "<span>" + esc(r.codecap) + "</span>" : "");
    }
    build();
    listeners.push(build);
    el("amode").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      amode = b.dataset.m;
      [].forEach.call(el("amode").children, (x) => {
        x.classList.toggle("on", x === b);
      });
      build();
    });
    el("smode").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      smode = b.dataset.m;
      [].forEach.call(el("smode").children, (x) => {
        x.classList.toggle("on", x === b);
      });
      build();
    });
    el("views").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      const v = VIEWS[b.dataset.v];
      goal = { t: v.t, p: v.p, r: v.r };
      touched = true;
      orbit = b.dataset.v === "orbit";
      [].forEach.call(el("views").children, (x) => {
        x.classList.toggle("on", x === b);
      });
    });
    canvas.addEventListener("pointerdown", (e) => {
      orbit = false;
      drag = { x: e.clientX, y: e.clientY, t: goal.t, p: goal.p };
      touched = true;
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
    });
    canvas.addEventListener("pointerup", () => {
      drag = null;
      canvas.style.cursor = "grab";
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        goal.r = Math.max(50, Math.min(260, goal.r * (1 + e.deltaY * 0.001)));
      },
      { passive: false },
    );
    const ray = new THREE.Raycaster(),
      mouse = new THREE.Vector2();
    ray.params.Points.threshold = 0.8;
    canvas.addEventListener("pointermove", (e) => {
      if (drag) {
        goal.t = drag.t - (e.clientX - drag.x) * 0.006;
        goal.p = Math.max(0.02, Math.min(1.52, drag.p - (e.clientY - drag.y) * 0.005));
        tip.hidden = true;
        return;
      }
      const b = canvas.getBoundingClientRect();
      mouse.x = ((e.clientX - b.left) / b.width) * 2 - 1;
      mouse.y = (-(e.clientY - b.top) / b.height) * 2 + 1;
      ray.setFromCamera(mouse, camera);
      const hit = pts && ray.intersectObject(pts)[0];
      if (!hit) {
        tip.hidden = true;
        return;
      }
      tipFor(tip, meta3[hit.index]);
      placeTip(wrap, tip, e);
    });
    canvas.addEventListener("pointerleave", () => {
      tip.hidden = true;
    });
    const v3 = new THREE.Vector3();
    (function frame() {
      if (!VIS(wrap)) {
        requestAnimationFrame(frame);
        return;
      }
      if (!touched) goal.t += 0.0007;
      if (orbit) goal.t += 0.004;
      sph.t += (goal.t - sph.t) * 0.12;
      sph.p += (goal.p - sph.p) * 0.12;
      sph.r += (goal.r - sph.r) * 0.12;
      place();
      renderer.render(scene, camera);
      labels.forEach((L) => {
        v3.copy(L.p).project(camera);
        if (v3.z > 1) {
          L.el.style.transform = "translate(-9999px,-9999px)";
          return;
        }
        L.el.style.transform = "translate(" + (((v3.x + 1) / 2) * W).toFixed(1) + "px," + (((1 - v3.y) / 2) * H).toFixed(1) + "px)";
      });
      requestAnimationFrame(frame);
    })();
  })();
