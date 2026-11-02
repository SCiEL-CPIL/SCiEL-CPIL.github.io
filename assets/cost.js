let cxm = "cost";
function costFig() {
  const W = 1160,
    H = 380,
    L = 84,
    Rr = 20,
    T = 14,
    B = 40,
    key = cxm === "cost" ? 1 : 2,
    all = [],
    dl = [];
  NAMES.forEach((n) => {
    R[n].engine.steps.forEach((s) => {
      if (s[key] > 0 && s[3] > 0) {
        all.push(s[key]);
        dl.push(dig(s[3]));
      }
    });
  });
  let x0 = Math.log10(Math.min.apply(null, all)),
    x1 = Math.log10(Math.max.apply(null, all));
  x0 = Math.floor(x0);
  x1 = Math.ceil(x1);
  let lo = Math.floor(Math.min.apply(null, dl) * 2) / 2;
  const hi = Math.ceil(Math.max.apply(null, dl) * 2) / 2;
  lo = Math.max(lo, 0.5);
  function X(v) {
    return L + ((Math.log10(v) - x0) / (x1 - x0)) * (W - L - Rr);
  }
  function Y(d) {
    return T + (1 - (Math.max(lo, d) - lo) / (hi - lo)) * (H - T - B);
  }
  const s = ['<svg viewBox="0 0 ' + W + " " + H + '">'];
  axisY(s, L, W, Rr, Y, lo, hi);
  for (let e = x0; e <= x1; e++) {
    const v = Math.pow(10, e);
    s.push(
      '<line y1="' +
        T +
        '" y2="' +
        (H - B) +
        '" x1="' +
        X(v) +
        '" x2="' +
        X(v) +
        '" stroke="#1C1C1E"/><text x="' +
        X(v) +
        '" y="' +
        (H - B + 20) +
        '" fill="#98989D" font-size="11" text-anchor="middle">' +
        (cxm === "cost" ? "$" + (v < 1 ? v.toFixed(-e) : v) : v) +
        "</text>",
    );
  }
  NAMES.forEach((n) => {
    const st = R[n].engine.steps.filter((s) => s[key] > 0 && s[3] > 0);
    let pb = -1;
    let d = "";
    st.forEach((q, k) => {
      const b = Math.max(pb, q[3]);
      d += (k ? "H" + X(q[key]).toFixed(1) + "V" : "M" + X(q[key]).toFixed(1) + " ") + Y(dig(b)).toFixed(1);
      pb = b;
    });
    if (pb > 1) {
      var lq = st[st.length - 1];
      s.push(
        '<text x="' +
          (X(lq[key]) - 8) +
          '" y="' +
          (Y(5) + 14) +
          '" fill="' +
          RUNC[n] +
          '" font-size="11" text-anchor="end">above the best known value</text>',
      );
    }
    var lq = st[st.length - 1];
    s.push(
      '<path d="' +
        d +
        '" fill="none" stroke="' +
        RUNC[n] +
        '" stroke-width="' +
        (n === cur ? 2.6 : 1.4) +
        '" opacity="' +
        (n === cur ? 1 : 0.7) +
        '"><title>' +
        label(n) +
        ": best " +
        f4(pb) +
        " for $" +
        lq[1].toFixed(2) +
        " and " +
        lq[2] +
        ' model calls</title></path><circle cx="' +
        X(lq[key]).toFixed(1) +
        '" cy="' +
        Y(dig(pb)).toFixed(1) +
        '" r="3.5" fill="' +
        RUNC[n] +
        '"><title>' +
        label(n) +
        ": best " +
        f4(pb) +
        " · $" +
        lq[1].toFixed(2) +
        " · " +
        lq[2] +
        " calls</title></circle>",
    );
  });
  s.push(
    '<text x="' +
      (W - Rr) +
      '" y="' +
      (H - 6) +
      '" fill="#98989D" font-size="11" text-anchor="end">' +
      (cxm === "cost" ? "cumulative model spend, log scale" : "model calls, log scale") +
      "</text></svg>",
  );
  el("cost").innerHTML = s.join("");
  el("costleg").innerHTML = NAMES.map(
    (n) =>
      '<span><i class="ln" style="background:' +
      RUNC[n] +
      '"></i>' +
      label(n) +
      " · $" +
      R[n].engine.steps[R[n].engine.steps.length - 1][1].toFixed(2) +
      "</span>",
  ).join("");
}
el("cxm").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  cxm = b.dataset.m;
  [].forEach.call(el("cxm").children, (x) => {
    x.classList.toggle("on", x === b);
  });
  costFig();
});
listeners.push(costFig);
costFig();
