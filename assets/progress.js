function editsFig() {
  const r = R[cur],
    rows = r.edits.filter((e) => e[0] > 0 && e[1] > 0),
    W = 620,
    H = 540,
    L = 56,
    B = 48,
    T = 14,
    Rr = 14;
  const v = rows.map((e) => [dig(e[0]), dig(e[1]), e[2]]),
    all = [].concat.apply(
      [],
      v.map((q) => [q[0], q[1]]),
    );
  let lo = Math.max(0, Math.min.apply(null, all)),
    hi = Math.max.apply(null, all);
  lo = Math.floor(lo * 2) / 2;
  hi = Math.ceil(hi * 2) / 2;
  function X(x) {
    return L + ((x - lo) / (hi - lo)) * (W - L - Rr);
  }
  function Y(y) {
    return T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
  }
  const above = v.filter((q) => q[1] > q[0]).length;
  let s = '<svg viewBox="0 0 ' + W + " " + H + '">';
  for (let g = Math.ceil(lo); g <= hi; g++) {
    s +=
      '<line x1="' +
      L +
      '" x2="' +
      (W - Rr) +
      '" y1="' +
      Y(g) +
      '" y2="' +
      Y(g) +
      '" stroke="#1C1C1E"/><line y1="' +
      T +
      '" y2="' +
      (H - B) +
      '" x1="' +
      X(g) +
      '" x2="' +
      X(g) +
      '" stroke="#1C1C1E"/><text x="' +
      (L - 10) +
      '" y="' +
      (Y(g) + 4) +
      '" fill="#98989D" font-size="11" text-anchor="end">' +
      sup(g) +
      '</text><text x="' +
      X(g) +
      '" y="' +
      (H - B + 18) +
      '" fill="#98989D" font-size="11" text-anchor="middle">' +
      sup(g) +
      "</text>";
  }
  s += '<line x1="' + X(lo) + '" y1="' + Y(lo) + '" x2="' + X(hi) + '" y2="' + Y(hi) + '" stroke="#98989D" stroke-dasharray="4 4"/>';
  v.forEach((q, k) => {
    s +=
      '<circle cx="' +
      X(q[0]).toFixed(1) +
      '" cy="' +
      Y(q[1]).toFixed(1) +
      '" r="3.8" fill="' +
      ic(q[2]) +
      '" stroke="#000" stroke-width=".8" opacity=".9"><title>deme ' +
      q[2] +
      ": parent " +
      (+rows[k][0]).toFixed(5) +
      " → child " +
      (+rows[k][1]).toFixed(5) +
      (q[1] > q[0] ? " · step up" : " · no gain") +
      "</title></circle>";
  });
  s +=
    '<text x="' +
    (L + (W - L) / 2) +
    '" y="' +
    (H - 6) +
    '" fill="#98989D" font-size="12" text-anchor="middle">parent: 1 − score</text><text x="16" y="' +
    (T + (H - T - B) / 2) +
    '" fill="#98989D" font-size="12" text-anchor="middle" transform="rotate(-90 16 ' +
    (T + (H - T - B) / 2) +
    ')">child: 1 − score</text>';
  s +=
    '<text x="' +
    (L + 12) +
    '" y="' +
    (T + 20) +
    '" fill="#F5F5F7" font-size="13" font-weight="600">' +
    above +
    " of " +
    v.length +
    " edits above the diagonal: π ≈ " +
    (above / (v.length || 1)).toFixed(2) +
    "</text></svg>";
  el("edits").innerHTML = s;
  el("s-edits").innerHTML =
    "In " +
    LBL(cur) +
    ", " +
    above +
    " of " +
    v.length +
    " edits beat their parent. Far from the frontier most edits help; near the best program the points crowd onto the diagonal, which is the regime where improvements are rare and recursive improvement has to pay.";
}
listeners.push(editsFig);
editsFig();
el("pis").innerHTML =
  '<div class="funnel">' +
  NAMES.map((n) => {
    const e = R[n].edits.filter((q) => q[0] > 0 && q[1] > 0),
      a = e.filter((q) => q[1] > q[0]).length,
      pi = a / (e.length || 1);
    const top = e.filter((q) => q[0] >= R[n].best - 0.01),
      ta = top.filter((q) => q[1] > q[0]).length,
      tp = top.length ? ta / top.length : 0;
    return (
      '<span class="s">' +
      label(n) +
      '</span><div><div class="b"><i style="width:' +
      pi * 100 +
      '%;background:#0A84FF"></i></div><div class="b" style="margin-top:4px"><i style="width:' +
      tp * 100 +
      '%;background:#E0637F"></i></div></div><span class="n">' +
      pi.toFixed(2) +
      '<br><span class="muted">' +
      tp.toFixed(2) +
      "</span></span>"
    );
  }).join("") +
  '</div><div class="legend"><span><i style="background:#0A84FF"></i>all edits</span><span><i style="background:#E0637F"></i>frontier</span></div>';

function ladder() {
  const r = R[cur],
    st = r.engine.steps,
    sw = r.engine.swm,
    W = 1160,
    H = 320,
    L = 60,
    Rr = 20,
    T = 14,
    mid = 92,
    B = 36;
  const mx = st[st.length - 1][0] || 1;
  function X(i) {
    return L + (i / mx) * (W - L - Rr);
  }
  const s = ['<svg viewBox="0 0 ' + W + " " + H + '">'],
    col = { base: "#48484D", swm: "#0A84FF", rl: "#3DAE7C" };
  s.push('<text x="' + (L - 10) + '" y="' + (T + 26) + '" fill="#98989D" font-size="11" text-anchor="end">planner</text>');
  st.forEach((q) => {
    const c = q[5].planner ? q[5].planner[0] : null;
    if (!c) return;
    s.push(
      '<rect x="' +
        (X(q[0]) - 1.2).toFixed(1) +
        '" y="' +
        (T + (c === "base" ? 18 : c === "swm" ? 8 : 0)) +
        '" width="2.4" height="' +
        (c === "base" ? 14 : 24) +
        '" fill="' +
        (col[c] || "#48484D") +
        '"/>',
    );
  });
  const vals = [];
  sw.forEach((q) => {
    [q[2], q[3], q[4]].forEach((v) => {
      if (v != null) vals.push(v);
    });
  });
  if (vals.length) {
    const lo = Math.min.apply(null, vals),
      hi = Math.min(Math.max.apply(null, vals), lo + 1.2);
    const Y = (v) => mid + ((Math.min(v, hi) - lo) / (hi - lo || 1)) * (H - mid - B);
    for (let g = 0; g <= 3; g++) {
      const v = lo + ((hi - lo) * g) / 3;
      s.push(
        '<line x1="' +
          L +
          '" x2="' +
          (W - Rr) +
          '" y1="' +
          Y(v) +
          '" y2="' +
          Y(v) +
          '" stroke="#1C1C1E"/><text x="' +
          (L - 10) +
          '" y="' +
          (Y(v) + 4) +
          '" fill="#98989D" font-size="11" text-anchor="end">' +
          v.toFixed(2) +
          "</text>",
      );
    }
    let ds = "",
      dr = "";
    sw.forEach((q, k) => {
      const ref = Math.min(q[3] != null ? q[3] : 9, q[4] != null ? q[4] : 9);
      ds += (k ? "L" : "M") + X(q[0]).toFixed(1) + " " + Y(q[2]).toFixed(1);
      dr += (k ? "L" : "M") + X(q[0]).toFixed(1) + " " + Y(ref).toFixed(1);
    });
    s.push(
      '<path d="' +
        dr +
        '" fill="none" stroke="#98989D" stroke-width="1.4" stroke-dasharray="4 4"/><path d="' +
        ds +
        '" fill="none" stroke="#0A84FF" stroke-width="1.8"/>',
    );
    sw.forEach((q) => {
      const ok = q[1] === "fit" && q[6] && q[6] !== "None";
      s.push(
        '<circle cx="' +
          X(q[0]).toFixed(1) +
          '" cy="' +
          Y(q[2]).toFixed(1) +
          '" r="4.5" fill="' +
          (ok ? "#0A84FF" : "#000") +
          '" stroke="#0A84FF" stroke-width="1.5"><title>step ' +
          q[0] +
          ": world model " +
          (q[2] != null ? q[2].toFixed(3) : "—") +
          ", cell means " +
          (q[3] != null ? q[3].toFixed(3) : "—") +
          ", base rate " +
          (q[4] != null ? q[4].toFixed(3) : "—") +
          " · " +
          (ok ? "acts" : "held back") +
          "</title></circle>",
      );
    });
    s.push('<text x="' + L + '" y="' + (mid - 10) + '" fill="#98989D" font-size="11">log loss, lower is better (top)</text>');
  }
  for (let t = 0; t <= mx; t += Math.max(50, Math.round(mx / 8 / 50) * 50))
    s.push('<text x="' + X(t) + '" y="' + (H - 8) + '" fill="#98989D" font-size="11" text-anchor="middle">' + t + "</text>");
  s.push("</svg>");
  el("ladder").innerHTML = s.join("");
  const c = {};
  st.forEach((q) => {
    const k = q[5].planner ? q[5].planner[0] : "none";
    c[k] = (c[k] || 0) + 1;
  });
  const fits = sw.filter((q) => q[1] === "fit").length;
  el("s-ladder").innerHTML =
    "In " +
    LBL(cur) +
    " the bandits planned " +
    (c.base || 0) +
    " steps and the world model " +
    (c.swm || 0) +
    (c.rl ? ", the RL policy " + c.rl : "") +
    "; the world model passed its check at " +
    fits +
    " of " +
    sw.length +
    " refits.";
}
listeners.push(ladder);
ladder();

function chart() {
  const r = R[cur],
    P = r.programs
      .filter((p) => p[4] != null && p[4] > 0)
      .map((p) => [p[6] || 0, p[2], p[4]])
      .sort((a, b) => a[0] - b[0]);
  const W = 1160;
  const H = 300;
  const L = 60;
  const Rr = 20;
  const T = 14;
  const B = 34;
  const maxIt =
    Math.max.apply(
      null,
      P.map((p) => p[0]),
    ) || 1;
  const isl = {};
  const pop = [];
  let pb = -1;
  const vals = P.map((p) => dig(p[2])).sort((a, b) => a - b),
    lo = Math.floor(vals[Math.floor(vals.length * 0.1)] * 2) / 2,
    hi = Math.ceil(dig(r.best) * 4) / 4;
  P.forEach((p) => {
    const b = isl[p[1]] || (isl[p[1]] = { best: -1, pts: [] });
    if (p[2] > b.best) {
      b.best = p[2];
      b.pts.push([p[0], p[2]]);
    }
    if (p[2] > pb) {
      pb = p[2];
      pop.push([p[0], pb]);
    }
  });
  function X(i) {
    return L + (i / maxIt) * (W - L - Rr);
  }
  function Y(d) {
    return T + (1 - (Math.max(lo, d) - lo) / (hi - lo || 1)) * (H - T - B);
  }
  function step(pts) {
    let d = "M" + X(pts[0][0]) + " " + Y(dig(pts[0][1]));
    for (let k = 1; k < pts.length; k++) d += "H" + X(pts[k][0]) + "V" + Y(dig(pts[k][1]));
    return d + "H" + X(maxIt);
  }
  const s = ['<svg viewBox="0 0 ' + W + " " + H + '">'];
  axisY(s, L, W, Rr, Y, lo, hi);
  for (let t = 0; t <= maxIt; t += Math.max(50, Math.round(maxIt / 8 / 50) * 50))
    s.push('<text x="' + X(t) + '" y="' + (H - 8) + '" fill="#98989D" font-size="11" text-anchor="middle">' + t + "</text>");
  Object.keys(isl).forEach((k) => {
    s.push('<path d="' + step(isl[k].pts) + '" fill="none" stroke="' + ic(+k) + '" stroke-width="1.4" opacity=".85"/>');
  });
  s.push('<path d="' + step(pop) + '" fill="none" stroke="#F5F5F7" stroke-width="2"/></svg>');
  el("chart").innerHTML = s.join("");
  el("chartleg").innerHTML =
    '<span><i class="ln" style="background:#F5F5F7"></i>population</span>' +
    Object.keys(isl)
      .map((k) => '<span><i class="ln" style="background:' + ic(+k) + '"></i>deme ' + k + "</span>")
      .join("");
  const who = {},
    ops = {};
  r.chain.forEach((c) => {
    who[c.island] = 1;
    ops[c.op] = (ops[c.op] || 0) + 1;
  });
  const topop = Object.keys(ops).sort((a, b) => ops[b] - ops[a])[0];
  el("s-steps").innerHTML =
    "In " +
    LBL(cur) +
    " the best program changed " +
    r.chain.length +
    " times, on " +
    Object.keys(who).length +
    " of " +
    Object.keys(isl).length +
    " demes. The other demes supply parents and material; the most common record move is " +
    (OPN[topop] || topop) +
    " (" +
    ops[topop] +
    " of " +
    r.chain.length +
    ").";
}
listeners.push(chart);
chart();
