(() => {
  const C = D.culture,
    y = C.out.y,
    ev = C.out.ev;
  function pc(v) {
    return Math.round(v * 100) + "%";
  }
  function pp(v) {
    return (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(1);
  }
  el("s-ct").innerHTML =
    "Each time a deme builds a prompt, SCiEL decides at random, with a logged probability, whether to include a finding from another deme. That makes the effect of culture estimable from the system’s own logs, without bias. Across " +
    XPL(C.runs + " runs") +
    " and " +
    C.n.toLocaleString("en") +
    " such decisions, a delivered finding raised the chance that the child beats its parent from " +
    pc(y[1]) +
    " to " +
    pc(y[0]) +
    " (" +
    pp(y[0] - y[1]) +
    " points, 95% interval " +
    pp(y[2]) +
    " to " +
    pp(y[3]) +
    "), and the chance of a new deme record from " +
    pc(ev[1]) +
    " to " +
    pc(ev[0]) +
    ". The effect is largest early in a run, when demes have the most to learn from each other.";
  const W = 560,
    H = 300,
    L = 150,
    Rr = 150,
    T = 40,
    s = ['<svg viewBox="0 0 ' + W + " " + H + '">'],
    x0 = 0.3,
    x1 = 0.7;
  function X(v) {
    return L + ((v - x0) / (x1 - x0)) * (W - L - Rr);
  }
  for (let g = 0.3; g <= 0.701; g += 0.1) {
    s.push(
      '<line x1="' +
        X(g) +
        '" x2="' +
        X(g) +
        '" y1="' +
        (T - 14) +
        '" y2="' +
        (H - 36) +
        '" stroke="#1C1C1E"/><text x="' +
        X(g) +
        '" y="' +
        (H - 16) +
        '" fill="#98989D" font-size="11" text-anchor="middle">' +
        Math.round(g * 100) +
        "%</text>",
    );
  }
  [
    ["child beats its parent", y],
    ["new deme record", ev],
  ].forEach((r, i) => {
    const yy = T + 40 + i * 100,
      v = r[1];
    s.push('<text x="' + (L - 16) + '" y="' + (yy + 4) + '" fill="#F5F5F7" font-size="13" text-anchor="end">' + r[0] + "</text>");
    s.push('<line x1="' + X(v[1]) + '" x2="' + X(v[0]) + '" y1="' + yy + '" y2="' + yy + '" stroke="#48484D" stroke-width="3"/>');
    s.push(
      '<circle cx="' +
        X(v[1]) +
        '" cy="' +
        yy +
        '" r="7" fill="#636366"><title>without a finding: ' +
        pc(v[1]) +
        '</title></circle><circle cx="' +
        X(v[0]) +
        '" cy="' +
        yy +
        '" r="8" fill="#E0637F"><title>with a finding from another deme: ' +
        pc(v[0]) +
        "</title></circle>",
    );
    s.push(
      '<text x="' +
        X(v[1]) +
        '" y="' +
        (yy - 14) +
        '" fill="#98989D" font-size="11" text-anchor="middle">' +
        pc(v[1]) +
        '</text><text x="' +
        X(v[0]) +
        '" y="' +
        (yy - 14) +
        '" fill="#E0637F" font-size="12" font-weight="600" text-anchor="middle">' +
        pc(v[0]) +
        "</text>",
    );
    s.push(
      '<text x="' +
        (W - Rr + 18) +
        '" y="' +
        (yy - 2) +
        '" fill="#F5F5F7" font-size="13" font-weight="600">' +
        pp(v[0] - v[1]) +
        ' pts</text><text x="' +
        (W - Rr + 18) +
        '" y="' +
        (yy + 15) +
        '" fill="#98989D" font-size="11">95%: ' +
        pp(v[2]) +
        " to " +
        pp(v[3]) +
        "</text>",
    );
  });
  s.push(
    '<g font-size="11"><circle cx="' +
      L +
      '" cy="14" r="5" fill="#636366"/><text x="' +
      (L + 10) +
      '" y="18" fill="#98989D">no finding</text><circle cx="' +
      (L + 96) +
      '" cy="14" r="5" fill="#E0637F"/><text x="' +
      (L + 106) +
      '" y="18" fill="#98989D">finding from another deme</text></g></svg>',
  );
  el("ctA").innerHTML = s.join("");
  const W2 = 560,
    H2 = 300,
    L2 = 130,
    R2 = 24,
    T2 = 24,
    d0 = -0.4,
    d1 = 0.45,
    t = ['<svg viewBox="0 0 ' + W2 + " " + H2 + '">'];
  function X2(v) {
    return L2 + ((Math.max(d0, Math.min(d1, v)) - d0) / (d1 - d0)) * (W2 - L2 - R2);
  }
  [-0.4, -0.2, 0, 0.2, 0.4].forEach((g) => {
    t.push(
      '<line x1="' +
        X2(g) +
        '" x2="' +
        X2(g) +
        '" y1="' +
        T2 +
        '" y2="' +
        (H2 - 36) +
        '" stroke="' +
        (g === 0 ? "#636366" : "#1C1C1E") +
        '"' +
        (g === 0 ? ' stroke-dasharray="4 4"' : "") +
        '/><text x="' +
        X2(g) +
        '" y="' +
        (H2 - 16) +
        '" fill="#98989D" font-size="11" text-anchor="middle">' +
        (g > 0 ? "+" : "") +
        Math.round(g * 100) +
        "</text>",
    );
  });
  t.push('<text x="' + (W2 - R2) + '" y="' + (H2 - 2) + '" fill="#98989D" font-size="11" text-anchor="end">difference, points</text>');
  const rows = C.phases.map((p) => [p[0], p[1] - p[2], p[3], p[4], p[5]]).concat([["all steps", y[0] - y[1], y[2], y[3], C.n]]);
  rows.forEach((r, i) => {
    const yy = T2 + 28 + i * 44,
      last = i === rows.length - 1;
    t.push(
      '<text x="' +
        (L2 - 14) +
        '" y="' +
        (yy + 4) +
        '" fill="' +
        (last ? "#F5F5F7" : "#C7C7CC") +
        '" font-size="12.5" font-weight="' +
        (last ? 600 : 400) +
        '" text-anchor="end">' +
        r[0] +
        "</text>",
    );
    t.push(
      '<line x1="' +
        X2(r[2]) +
        '" x2="' +
        X2(r[3]) +
        '" y1="' +
        yy +
        '" y2="' +
        yy +
        '" stroke="#E0637F" stroke-width="2" opacity=".7"/><rect x="' +
        (X2(r[1]) - 5) +
        '" y="' +
        (yy - 5) +
        '" width="10" height="10" fill="#E0637F"><title>' +
        r[0] +
        ": " +
        pp(r[1]) +
        " points, 95% " +
        pp(r[2]) +
        " to " +
        pp(r[3]) +
        ", " +
        r[4] +
        " steps</title></rect>",
    );
  });
  const yr = T2 + 28 + rows.length * 44;
  t.push('<text x="' + (L2 - 14) + '" y="' + (yr + 4) + '" fill="#C7C7CC" font-size="12.5" text-anchor="end">each run</text>');
  C.per_run.forEach((r, i) => {
    const jit = (((i * 37) % 11) - 5) * 1.6;
    t.push(
      '<circle cx="' +
        X2(r[1]).toFixed(1) +
        '" cy="' +
        (yr + jit).toFixed(1) +
        '" r="' +
        (2.5 + Math.sqrt(r[2]) / 2.2).toFixed(1) +
        '" fill="#98989D" opacity=".55"><title>' +
        esc(r[0]) +
        ": " +
        pp(r[1]) +
        " points over " +
        r[2] +
        " steps</title></circle>",
    );
  });
  t.push("</svg>");
  el("ctB").innerHTML = t.join("");
})();

function funnel() {
  const r = R[cur],
    x = r.engine.research,
    ph = x.phases || {},
    st = x.statuses || {},
    co = x.conclusions || {},
    ac = x.actions || {};
  if (!x.cells) {
    el("funnel").innerHTML = '<span class="s">Research cells were off in this run.</span><span></span><span></span>';
    el("s-funnel").innerHTML = "Circle packing ran without research cells; pick an autocorrelation run to see the funnel.";
    return;
  }
  const rose = r.transfers.filter((t) => t.rose).length;
  const rows = [
    ["cells opened", x.cells, "#48484D"],
    ["experiments planned", x.decisions, "#48484D"],
    ["tools written", ac.create || 0, "#0A84FF"],
    ["concluded supported or refuted", (co.supported || 0) + (co.refuted || 0), "#A66BE0"],
    ["findings shared", ph.sharing_completed || 0, "#3DAE7C"],
    ["adopted for trials", st.adopted_for_trials || 0, "#3DAE7C"],
    ["reached a child on another deme", r.transfers.length, "#E0637F"],
    ["child improved", rose, "#E0637F"],
  ];
  const TIP = {
    "cells opened": "standing questions opened by demes",
    "experiments planned": "model steps that planned an experiment for a cell",
    "tools written": "experiments that produced an executable tool",
    "tools ran": "tool invocations in the sandbox",
    "ran without error": "invocations that returned a result",
    "concluded supported or refuted": "steps that reached a verdict on their question",
    "findings shared": "conclusions shared with other demes",
    "adopted for trials": "shared findings other demes took on trial",
    "reached a child on another deme": "a finding delivered into a child prompt on another deme",
    "child improved": "that child beat its parent",
  };
  const mx = rows[0][1] || 1;
  el("funnel").innerHTML = rows
    .map(
      (q) =>
        '<span class="s" title="' +
        TIP[q[0]] +
        '">' +
        q[0] +
        '</span><div class="b"><i style="width:' +
        Math.max(q[1] ? 1.5 : 0, (q[1] / mx) * 100) +
        "%;background:" +
        q[2] +
        '"></i></div><span class="n">' +
        q[1] +
        "</span>",
    )
    .join("");

  el("s-funnel").innerHTML =
    "A research cell is a standing question an individual keeps open. Each step plans an experiment, writes a tool, runs it in a sandbox and concludes; what survives is shared and tried by others. In " +
    label(cur) +
    " " +
    x.cells +
    " cells planned " +
    x.decisions +
    " experiments and wrote " +
    (ac.create || 0) +
    " tools; " +
    ((co.supported || 0) + (co.refuted || 0)) +
    " experiments reached a verdict, and " +
    (st.adopted_for_trials || 0) +
    " findings were adopted for trials by other demes.";
}
listeners.push(funnel);
funnel();
