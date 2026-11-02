function diffLines(a, b) {
  a = a.split("\n");
  b = b.split("\n");
  const n = a.length,
    m = b.length;
  if (n * m > 6e6) return b.map((x) => ["a", x]);
  const dp = new Array(n + 1);
  for (var i = 0; i <= n; i++) dp[i] = new Int32Array(m + 1);
  for (i = n - 1; i >= 0; i--) for (var j = m - 1; j >= 0; j--) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  i = 0;
  j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push(["c", a[i]]);
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push(["r", a[i]]);
      i++;
    } else {
      out.push(["a", b[j]]);
      j++;
    }
  }
  while (i < n) out.push(["r", a[i++]]);
  while (j < m) out.push(["a", b[j++]]);
  const keep = out.map(() => false);
  out.forEach((o, k) => {
    if (o[0] !== "c") for (let q = Math.max(0, k - 3); q <= Math.min(out.length - 1, k + 3); q++) keep[q] = true;
  });
  const res = [];
  let gap = false;
  out.forEach((o, k) => {
    if (keep[k]) {
      res.push(o);
      gap = false;
    } else if (!gap) {
      res.push(["h", "  ⋯"]);
      gap = true;
    }
  });
  return res;
}

(() => {
  const M = D.demo;
  if (!M || !M.steps.length) return;
  el("demonote").textContent = "Diffs from an early run on the " + M.task + " (best " + M.best.toFixed(4) + "); the code of the runs above stays private.";
  el("recs").innerHTML = M.steps
    .map(
      (c, k) =>
        '<tr class="pick" data-k="' +
        k +
        '"><td class="num">' +
        c.it +
        '</td><td class="num">' +
        (+c.score).toFixed(5) +
        '</td><td><span class="tag dot" style="--fc:' +
        (OPC[c.op] || "#8E8E93") +
        '">' +
        (OPN[c.op] || c.op) +
        '</span></td><td class="idea">' +
        esc(c.idea.slice(0, 140)) +
        (c.idea.length > 140 ? "…" : "") +
        "</td></tr>",
    )
    .join("");
  const rows = el("recs").children;
  function pick(k) {
    const c = M.steps[k];
    [].forEach.call(rows, (x, i) => {
      x.classList.toggle("sel", i === k);
    });
    el("dtitle").innerHTML =
      "<b>Attempt " +
      c.it +
      '</b> <span class="muted">· ' +
      (+c.before).toFixed(5) +
      ' → <span class="up">' +
      (+c.score).toFixed(5) +
      "</span> · " +
      (OPN[c.op] || c.op) +
      "</span>";
    el("diff").innerHTML = c.diff
      .map((o) => '<span class="' + o[0] + '">' + esc((o[0] === "a" ? "+ " : o[0] === "r" ? "- " : o[0] === "h" ? "" : "  ") + o[1]) + "</span>")
      .join("");
  }
  [].forEach.call(rows, (tr) => {
    tr.onclick = () => {
      pick(+tr.dataset.k);
    };
  });
  let big = 0;
  M.steps.forEach((c, k) => {
    if (c.score - c.before > M.steps[big].score - M.steps[big].before) big = k;
  });
  pick(big);
})();

el("multi").innerHTML = NAMES.map((n) => {
  const r = R[n],
    P = r.programs
      .filter((p) => p[4] > 0)
      .map((p) => [p[6] || 0, p[4], p[7]])
      .sort((a, b) => a[0] - b[0]);
  const W = 360;
  const H = 160;
  const L = 8;
  const T = 8;
  const B = 22;
  const maxIt = P[P.length - 1][0] || 1;
  let pb = -1;
  const pts = [];
  P.forEach((p) => {
    if (p[1] > pb) {
      pb = p[1];
      pts.push([p[0], pb]);
    }
  });
  const lo = Math.min(pts[0][1], r.best - 0.05),
    hi = r.best;
  function X(i) {
    return L + (i / maxIt) * (W - 2 * L);
  }
  function Y(v) {
    return T + (1 - (Math.max(lo, v) - lo) / (hi - lo)) * (H - T - B);
  }
  let d = "M" + X(pts[0][0]) + " " + Y(pts[0][1]);
  pts.forEach((q) => {
    d += "H" + X(q[0]) + "V" + Y(q[1]);
  });
  d += "H" + X(maxIt);
  const dots = P.map((p) =>
    p[1] > lo
      ? '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="1.4" fill="' + (OPC[p[2]] || "#8E8E93") + '" opacity=".55"/>'
      : "",
  ).join("");
  return (
    '<div><p class="t"><b>' +
    LBL(n) +
    "</b> · best " +
    f4(r.best) +
    '</p><svg viewBox="0 0 ' +
    W +
    " " +
    H +
    '"><line x1="' +
    L +
    '" x2="' +
    (W - L) +
    '" y1="' +
    (H - B) +
    '" y2="' +
    (H - B) +
    '" stroke="#2C2C2E"/>' +
    dots +
    '<path d="' +
    d +
    '" fill="none" stroke="#F5F5F7" stroke-width="1.6"/><text x="' +
    L +
    '" y="' +
    (H - 5) +
    '" fill="#98989D" font-size="10">0</text><text x="' +
    (W - L) +
    '" y="' +
    (H - 5) +
    '" fill="#98989D" font-size="10" text-anchor="end">' +
    maxIt +
    " attempts</text></svg></div>"
  );
}).join("");

el("s-limits").innerHTML =
  'How fast the ratchet turns across generations, and whether the procedures that create culture improve too. How the noise radius ε sets what counts as a real step, measured per task before a run. How far the mechanism carries beyond program discovery, to the 959 task packages of the <a href="https://research-arena.small-pine-3af7.workers.dev" target="_blank" rel="noopener">Empira arena</a>, an independent arena for self-improving systems on the stream of science.';
