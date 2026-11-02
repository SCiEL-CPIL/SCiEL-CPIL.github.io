if (window.THREE)
  (() => {
    const WD = makeWorld();
    const { wrap, cv, ovm, rd, sc, cam, V, XL, sp, gl, dim, cols, plc, at, panels, MLAB, discT, ringT, DL, MODES, MCOL0 } = mechanismScene(WD);
    const B = 200;
    const opt = { policy: true, world: true, culture: true, research: true };
    const MCOL = MCOL0;
    const R4 = {};
    let runSeed = 1007;
    let looping = true;
    let loopT = null;
    let drg = null;
    let touched = false;
    function arc(a, b, hgt, n) {
      const pa = at(a, 0.4),
        pb = at(b, 0.4),
        out = [];
      for (let k = 0; k <= n; k++) {
        const t = k / n,
          p = pa.clone().lerp(pb, t);
        p.y += Math.sin(Math.PI * t) * hgt;
        out.push(p);
      }
      return out;
    }
    function trail(a, b) {
      const out = [];
      for (let k = 0; k <= 6; k++) {
        const t = k / 6;
        out.push(at([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], 0.45));
      }
      return out;
    }
    function buildMode(mode) {
      const old = R4[mode];
      if (old) {
        sc.remove(old.g);
        old.g.traverse((o) => {
          if (o.geometry) o.geometry.dispose();
          if (o.material) o.material.dispose();
        });
      }
      const run = runMode(WD, mode, B, runSeed, mode === "sciel" ? opt : {}),
        g = new THREE.Group();
      g.visible = false;
      sc.add(g);
      run.objs = [];
      run.ev.forEach((e) => {
        const c = new THREE.Color(mode === "bestn" ? "#C7C7CC" : mode === "climber" ? "#F5F5F7" : MCOL[e.i % 5]);
        let obj;
        if (e.k === "screen") {
          obj = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringT, color: 0x98989d, transparent: true, opacity: 0.7, depthWrite: false }));
          obj.position.copy(at(e.p, 0.5));
          obj.scale.set(1.3, 1.3, 1);
        } else if (e.k === "chord") {
          obj = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(arc(e.from, e.p, 7, 30)),
            e.acc
              ? new THREE.LineBasicMaterial({ color: 0xe0637f })
              : new THREE.LineDashedMaterial({ color: 0xe0637f, dashSize: 0.7, gapSize: 0.6, transparent: true, opacity: 0.7 }),
          );
          if (!e.acc) obj.computeLineDistances();
        } else if (e.k === "jump") {
          obj = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(arc(e.from, e.p, 5, 24)),
            new THREE.LineDashedMaterial({ color: 0xa66be0, dashSize: 0.5, gapSize: 0.4 }),
          );
          obj.computeLineDistances();
        } else if (e.acc && mode !== "bestn") {
          obj = new THREE.Line(new THREE.BufferGeometry().setFromPoints(trail(e.from, e.p)), new THREE.LineBasicMaterial({ color: c }));
        } else {
          obj = new THREE.Sprite(
            new THREE.SpriteMaterial({ map: discT, color: c, transparent: true, opacity: mode === "bestn" && e.acc ? 1 : 0.45, depthWrite: false }),
          );
          obj.position.copy(at(e.p, 0.4));
          const s = mode === "bestn" && e.acc ? 2 : 0.9;
          obj.scale.set(s, s, 1);
        }
        obj.visible = false;
        g.add(obj);
        run.objs.push(obj);
      });
      const heads = [],
        nl = mode === "bestn" || mode === "climber" ? 1 : 5;
      for (let i = 0; i < nl; i++) {
        const cc = new THREE.Color(mode === "bestn" ? "#C7C7CC" : mode === "climber" ? "#F5F5F7" : MCOL[i]);
        const hd = new THREE.Sprite(new THREE.SpriteMaterial({ map: discT, color: cc, transparent: true, depthWrite: false }));
        hd.scale.set(2.6, 2.6, 1);
        g.add(hd);
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(0.9, 1, 48),
          new THREE.MeshBasicMaterial({ color: cc, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }),
        );
        ring.rotation.x = -Math.PI / 2;
        g.add(ring);
        const fill = new THREE.Mesh(
          new THREE.CircleGeometry(1, 48),
          new THREE.MeshBasicMaterial({ color: cc, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }),
        );
        fill.rotation.x = -Math.PI / 2;
        g.add(fill);
        heads.push({ sp: hd, ring: ring, fill: fill });
      }
      R4[mode] = { run: run, g: g, heads: heads };
    }
    function buildAll() {
      MODES.forEach((m) => {
        buildMode(m[0]);
      });
      show(+el("mT").value);
    }
    function show(t) {
      MODES.forEach((m, mi) => {
        const mode = m[0],
          o = R4[mode];
        if (!o) return;
        const evs = o.run.ev;
        let best = 0;
        const state = {};
        for (let k = 0; k < evs.length; k++) {
          const e = evs[k];
          o.run.objs[k].visible = e.t <= t && !(e.k === "screen" && (e.t >= t || e.t < t - 20));
          if (e.t <= t && e.v != null && e.v > best) best = e.v;
          if (e.t <= t && e.pol) state[e.pol.i] = e.pol;
          if (e.t <= t && e.acc && e.k !== "screen" && mode !== "bestn") state["h" + e.i] = e.p;
        }
        o.heads.forEach((h, i) => {
          const p = mode === "bestn" ? WD.seed : state["h" + i] || (state[i] && state[i].c) || WD.seed,
            po = state[i];
          h.sp.position.copy(at(p, 1.1));
          const sg = po ? po.s : 0.09,
            mu = po ? po.mu : [0, 0],
            rr = Math.max(1.4, ((sg * XL) / 2) * 1.6),
            cp = at([p[0] + mu[0], p[1] + mu[1]], 0.6);
          const vis = mode !== "bestn" && t > 0;
          h.ring.visible = h.fill.visible = vis;
          h.ring.position.copy(cp);
          h.fill.position.copy(cp);
          const stretch = 1 + Math.min(2.5, Math.hypot(mu[0], mu[1]) * 25),
            ang = Math.atan2(-mu[1], mu[0]);
          h.ring.scale.set(rr * stretch, rr, 1);
          h.fill.scale.set(rr * stretch, rr, 1);
          h.ring.rotation.z = ang;
          h.fill.rotation.z = ang;
        });

        const dl = DL[mode];
        if (mode === "bestn") {
          let bi = -1,
            bv = -1;
          for (let q = 0; q < evs.length; q++) {
            if (evs[q].t <= t && evs[q].v > bv) {
              bv = evs[q].v;
              bi = q;
            }
          }
          dl.best.on = bi >= 0;
          if (bi >= 0) dl.best.p.copy(at(evs[bi].p, 3));
        }
        if (mode === "climber") {
          const hp = state.h0 || WD.seed;
          let stall = 0;
          for (let q2 = evs.length - 1; q2 >= 0; q2--) {
            const e2 = evs[q2];
            if (e2.t > t) continue;
            if (e2.acc) break;
            stall++;
          }
          dl.head.p.copy(at(hp, 3.4));
          dl.head.w = 0;
          dl.head.el.textContent = stall >= 25 ? "climber: stuck on a local bump" : "climber";
          dl.head.on = t > 0;
        }
        if (mode === "swarm")
          dl.heads.forEach((L, i) => {
            const hp = state["h" + i] || (state[i] && state[i].c);
            L.on = !!hp && t > 0;
            if (hp) L.p.copy(at(hp, 3.2));
          });
        if (mode === "sciel") {
          dl.heads.forEach((L, i) => {
            const hp = state["h" + i] || (state[i] && state[i].c);
            L.on = !!hp && t > 0 && t < B * 0.55;
            if (hp) L.p.copy(at(hp, 3.2));
          });
          let lc = null,
            ls = null,
            lj = null;
          for (let q3 = 0; q3 < evs.length; q3++) {
            const e3 = evs[q3];
            if (e3.t > t) break;
            if (e3.k === "chord" && e3.acc && e3.t > t - 40) lc = e3;
            if (e3.k === "screen" && e3.t < t && e3.t >= t - 6) ls = e3;
            if (e3.k === "jump" && e3.t > t - 30) lj = e3;
          }
          dl.chord.on = !!lc;
          if (lc) {
            const mp = [(lc.from[0] + lc.p[0]) / 2, (lc.from[1] + lc.p[1]) / 2];
            dl.chord.p.copy(at(mp, 8.5));
          }
          dl.screen.on = !!ls && best < 0.95;
          if (ls) dl.screen.p.copy(at(ls.p, 2.4));
          dl.jump.on = !!lj;
          if (lj) {
            const mj = [(lj.from[0] + lj.p[0]) / 2, (lj.from[1] + lj.p[1]) / 2];
            dl.jump.p.copy(at(mj, 6.5));
          }
          const p0 = state[0];
          dl.policy.on = !!p0 && t > 8 && best < 0.95;
          if (p0) dl.policy.p.copy(at([p0.c[0] + p0.mu[0], p0.c[1] + p0.mu[1] - 0.07], 1.2));
        }
        panels[mi].querySelector(".mb").innerHTML = "best <b>" + best.toFixed(3) + "</b>" + (best > 0.95 ? ' · <b style="color:#3DAE7C">summit</b>' : "");
      });
      el("mTv").textContent = "run " + (runSeed - 1006) + " · evaluation " + t + " of " + B;
    }
    el("mcomp").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      const c = b.dataset.c;
      opt[c] = !opt[c];
      b.classList.toggle("on", opt[c]);
      buildMode("sciel");
      play();
    });
    el("mT").addEventListener("input", function () {
      looping = false;
      stop();
      show(+this.value);
    });
    let raf = null;
    function stop() {
      if (raf) {
        cancelAnimationFrame(raf);
        raf = null;
      }
      clearTimeout(loopT);
      el("mplay").textContent = looping ? "Pause" : "Play";
    }
    function play() {
      if (raf) cancelAnimationFrame(raf);
      clearTimeout(loopT);
      looping = true;
      el("mplay").textContent = "Pause";
      const t0 = performance.now();
      (function st(now) {
        const t = Math.min(B, Math.round(((now - t0) / 9000) * B));
        el("mT").value = t;
        show(t);
        if (t < B) {
          raf = requestAnimationFrame(st);
          return;
        }
        raf = null;
        loopT = setTimeout(() => {
          if (!looping) return;
          runSeed++;
          buildAll();
          play();
        }, 2600);
      })(t0);
    }
    el("mplay").addEventListener("click", () => {
      if (looping) {
        looping = false;
        stop();
      } else play();
    });
    buildAll();
    const io = new IntersectionObserver(
      (es) => {
        es.forEach((en) => {
          if (en.isIntersecting && !wrap.dataset.played) {
            wrap.dataset.played = 1;
            play();
          }
        });
      },
      { threshold: 0.3 },
    );
    io.observe(wrap);
    setTimeout(() => {
      const rows = [
        ["One call · best of N", "bestn", {}],
        ["One climber", "climber", {}],
        ["Swarm", "swarm", {}],
        ["SCiEL", "sciel", {}],
        ["SCiEL without the policy", "sciel", { policy: false }],
        ["SCiEL without the world model", "sciel", { world: false }],
        ["SCiEL without culture", "sciel", { culture: false }],
        ["SCiEL without research cells", "sciel", { research: false }],
      ];
      let out =
        '<span class="hd">30 runs each, 200 evaluations</span><span class="hd">mean best score</span><span class="hd" style="text-align:right">reached the summit</span>';
      rows.forEach((r) => {
        let s = 0,
          h = 0;
        for (let k = 0; k < 30; k++) {
          const x = runMode(WD, r[1], B, 500 + k, r[2]);
          s += x.best;
          h += x.best > 0.95;
        }
        const m = s / 30,
          isS = r[0] === "SCiEL";
        out +=
          '<span class="s" style="' +
          (isS ? "color:var(--ink);font-weight:600" : "") +
          '">' +
          r[0] +
          '</span><div class="b"><i style="width:' +
          m * 100 +
          "%;background:" +
          (r[1] === "sciel" ? (isS ? "#0A84FF" : "rgba(10,132,255,.45)") : "#48484D") +
          '"></i></div><span class="n">' +
          m.toFixed(2) +
          " · " +
          h +
          "/30</span>";
      });
      el("mcompare").innerHTML = out;
    }, 300);
    cv.addEventListener("pointerdown", (e) => {
      drg = { x: e.clientX, y: e.clientY, t: gl.t, p: gl.p };
      touched = true;
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener("pointerup", () => {
      drg = null;
    });
    cv.addEventListener("pointermove", (e) => {
      if (!drg) return;
      gl.t = drg.t - (e.clientX - drg.x) * 0.006;
      gl.p = Math.max(0.12, Math.min(1.45, drg.p - (e.clientY - drg.y) * 0.005));
    });
    cv.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        gl.r = Math.max(70, Math.min(240, gl.r * (1 + e.deltaY * 0.001)));
      },
      { passive: false },
    );
    (function fr() {
      if (!VIS(wrap)) {
        requestAnimationFrame(fr);
        return;
      }
      if (!touched) gl.t += 0.0005;
      sp.t += (gl.t - sp.t) * 0.12;
      sp.p += (gl.p - sp.p) * 0.12;
      sp.r += (gl.r - sp.r) * 0.12;
      plc();
      const c = cols();
      MODES.forEach((m, i) => {
        MODES.forEach((n) => {
          if (R4[n[0]]) R4[n[0]].g.visible = n[0] === m[0];
        });
        const x = (i % c) * dim.PW,
          y = dim.MH - (Math.floor(i / c) + 1) * dim.PH;
        rd.setViewport(x, y, dim.PW, dim.PH);
        rd.setScissor(x, y, dim.PW, dim.PH);
        rd.render(sc, cam);
      });
      const v3 = new V();
      MLAB.forEach((L) => {
        if (!L.on) {
          L.el.style.transform = "translate(-9999px,-9999px)";
          return;
        }
        v3.copy(L.p).project(cam);
        if (v3.z > 1 || Math.abs(v3.x) > 1.05 || Math.abs(v3.y) > 1.05) {
          L.el.style.transform = "translate(-9999px,-9999px)";
          return;
        }
        const ox = (L.pi % c) * dim.PW,
          oy = Math.floor(L.pi / c) * dim.PH,
          hw = (L.w || (L.w = L.el.offsetWidth || 80)) / 2 + 8,
          px = Math.max(ox + hw, Math.min(ox + dim.PW - hw, ox + ((v3.x + 1) / 2) * dim.PW)),
          py = Math.max(oy + 58, oy + ((1 - v3.y) / 2) * dim.PH);
        L.el.style.transform = "translate(" + px.toFixed(1) + "px," + py.toFixed(1) + "px)";
      });
      requestAnimationFrame(fr);
    })();
  })();
