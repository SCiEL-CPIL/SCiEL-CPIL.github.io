function geodesic(p, q, steps) {
  steps = steps || 18;
  const x1 = p[0],
    y1 = p[1],
    x2 = q[0],
    y2 = q[1],
    det = x1 * y2 - x2 * y1,
    out = [];
  if (Math.abs(det) < 1e-9) {
    for (var t = 0; t <= steps; t++) out.push([x1 + ((x2 - x1) * t) / steps, y1 + ((y2 - y1) * t) / steps]);
    return out;
  }
  const k1 = (x1 * x1 + y1 * y1 + 1) / 2,
    k2 = (x2 * x2 + y2 * y2 + 1) / 2,
    cx = (k1 * y2 - k2 * y1) / det,
    cy = (x1 * k2 - x2 * k1) / det,
    rad = Math.hypot(x1 - cx, y1 - cy),
    a1 = Math.atan2(y1 - cy, x1 - cx),
    a2 = Math.atan2(y2 - cy, x2 - cx),
    d = ((((a2 - a1 + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
  for (t = 0; t <= steps; t++) out.push([cx + rad * Math.cos(a1 + (d * t) / steps), cy + rad * Math.sin(a1 + (d * t) / steps)]);
  return out;
}
function diskModel(r) {
  if (r._disk) return r._disk;
  const pos = {},
    dots = {},
    texts = [];
  r.disk.forEach((i) => {
    if (i.t === "dot" && i.id) {
      pos[i.id] = [2 * i.x - 1, 2 * i.y - 1];
      dots[i.id] = i;
    } else if (i.t === "text") texts.push(i);
  });
  const by = byId(r),
    best = {};
  r.programs.forEach((p) => {
    if (p[4] != null && (!(p[2] in best) || p[4] > by[best[p[2]]][4])) best[p[2]] = p[0];
  });
  const cellProg = {};
  r.cells.forEach((c) => {
    cellProg[c.cell] = c.program;
  });
  const chords = r.transfers
    .map((t) => {
      const o = cellProg[r.fcell[t.finding]] || best[t.from_island];
      return { from: o, to: t.child, rose: t.rose };
    })
    .filter((c) => pos[c.from] && pos[c.to]);
  return (r._disk = { pos: pos, dots: dots, texts: texts, chords: chords });
}
function diskSvg(r, S, tmax) {
  const m = diskModel(r),
    by = byId(r),
    pad = 16,
    s = ['<svg viewBox="' + -pad + " " + -pad + " " + (S + 2 * pad) + " " + (S + 2 * pad) + '" role="img" aria-label="Culture disk">'];
  function X(x) {
    return ((x + 1) / 2) * S;
  }
  function Y(y) {
    return (1 - (y + 1) / 2) * S;
  }
  function path(pts) {
    return pts.map((q, k) => (k ? "L" : "M") + X(q[0]).toFixed(1) + " " + Y(q[1]).toFixed(1)).join("");
  }
  function shown(id) {
    const p = by[id];
    return p && (tmax == null || (p[6] || 0) <= tmax);
  }
  s.push('<circle cx="' + S / 2 + '" cy="' + S / 2 + '" r="' + S / 2 + '" fill="#0B0B0C" stroke="#2C2C2E"/>');
  m.texts.forEach((i) => {
    s.push(
      '<text x="' +
        i.x * S +
        '" y="' +
        (1 - i.y) * S +
        '" fill="#98989D" font-size="' +
        (S / 52).toFixed(1) +
        '" text-anchor="middle" dominant-baseline="middle">' +
        esc(String(i.s).replace(/^I(\d+)$/, "D$1")) +
        "</text>",
    );
  });
  [1, 3, 6].forEach((g) => {
    const rr = (Math.tanh(0.42 * g + 0.25) * S) / 2;
    s.push('<circle cx="' + S / 2 + '" cy="' + S / 2 + '" r="' + rr.toFixed(1) + '" fill="none" stroke="#2C2C2E" stroke-dasharray="2 5"/>');
  });
  Object.keys(m.pos).forEach((id) => {
    if (!shown(id)) return;
    const p = by[id],
      src = p[1] && m.pos[p[1]] ? m.pos[p[1]] : [0, 0];
    s.push('<path data-l="' + id + '" d="' + path(geodesic(src, m.pos[id])) + '" fill="none" stroke="' + ic(p[2]) + '" stroke-width="1" opacity=".55"/>');
  });
  let chordLab = null;
  m.chords.forEach((c) => {
    if (!shown(c.to)) return;
    const g = geodesic(m.pos[c.from], m.pos[c.to], 40);
    s.push(
      "<path" +
        (c.rose && !chordLab ? ' data-chord="1"' : "") +
        ' d="' +
        path(g) +
        '" fill="none" stroke="#E0637F" stroke-width="2"' +
        (c.rose ? "" : ' stroke-dasharray="5 5"') +
        "/>",
    );
    if (c.rose && !chordLab) chordLab = 1;
  });

  Object.keys(m.dots).forEach((id) => {
    if (!shown(id)) return;
    const i = m.dots[id],
      rr = Math.max(1.6, i.r * S);
    s.push(
      '<circle cx="' +
        (i.x * S).toFixed(1) +
        '" cy="' +
        ((1 - i.y) * S).toFixed(1) +
        '" r="' +
        rr.toFixed(1) +
        '" fill="' +
        (i.fill ? i.c : "#0B0B0C") +
        '" stroke="' +
        i.c +
        '" stroke-width="1.2" data-id="' +
        id +
        '"/>',
    );
  });
  let bi = null;
  Object.keys(m.dots).forEach((id) => {
    if (shown(id) && by[id][4] != null && (!bi || by[id][4] > by[bi][4])) bi = id;
  });
  if (bi) {
    const bd = m.dots[bi],
      bx = bd.x * S,
      byy = (1 - bd.y) * S,
      rr = Math.max(1.6, bd.r * S) + 6,
      lft = bx < S / 2;
    s.push(
      '<circle cx="' +
        bx.toFixed(1) +
        '" cy="' +
        byy.toFixed(1) +
        '" r="' +
        rr.toFixed(1) +
        '" fill="none" stroke="#F5F5F7" stroke-width="1.2"/><text x="' +
        (bx + (lft ? rr + 6 : -rr - 6)).toFixed(1) +
        '" y="' +
        (byy + 4).toFixed(1) +
        '" fill="#F5F5F7" font-size="' +
        (S / 48).toFixed(1) +
        '" font-weight="600" text-anchor="' +
        (lft ? "start" : "end") +
        '">best ' +
        f4(by[bi][4]) +
        "</text>",
    );
  }
  s.push(
    '<circle cx="' +
      S / 2 +
      '" cy="' +
      S / 2 +
      '" r="2.5" fill="#E8E8EA"/><text x="' +
      (S / 2 + 6) +
      '" y="' +
      (S / 2 + 14) +
      '" fill="#636366" font-size="' +
      (S / 60).toFixed(1) +
      '">seed</text>',
  );
  return s.join("") + "</svg>";
}

function codeSvg(r, W, H) {
  const c = r.coords.code;
  const by = byId(r);
  const L = 48;
  const B = 40;
  const T = 8;
  const Rr = 8;
  let s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Code map">';
  function X(x) {
    return L + x * (W - L - Rr);
  }
  function Y(y) {
    return T + (1 - y) * (H - T - B);
  }
  const sc = r.programs.filter((p) => c[p[0]] && p[4] != null).map((p) => p[4]),
    lo = Math.min.apply(null, sc),
    hi = Math.max.apply(null, sc);
  s +=
    '<defs><radialGradient id="bl"><stop offset="0" stop-color="#0A84FF" stop-opacity="1"/><stop offset="1" stop-color="#0A84FF" stop-opacity="0"/></radialGradient></defs>';
  s += '<rect x="' + L + '" y="' + T + '" width="' + (W - L - Rr) + '" height="' + (H - T - B) + '" fill="#0B0B0C" stroke="#2C2C2E"/>';
  Object.keys(c).forEach((id) => {
    const p = by[id];
    if (!p || p[4] == null) return;
    s +=
      '<circle cx="' +
      X(c[id][0]).toFixed(1) +
      '" cy="' +
      Y(c[id][1]).toFixed(1) +
      '" r="' +
      W * 0.045 +
      '" fill="url(#bl)" opacity="' +
      (0.05 + 0.3 * Math.pow((p[4] - lo) / (hi - lo || 1), 3)).toFixed(3) +
      '"/>';
  });
  Object.keys(c).forEach((id) => {
    const p = by[id];
    if (p && p[1] && c[p[1]])
      s +=
        '<line data-l="' +
        id +
        '" x1="' +
        X(c[p[1]][0]).toFixed(1) +
        '" y1="' +
        Y(c[p[1]][1]).toFixed(1) +
        '" x2="' +
        X(c[id][0]).toFixed(1) +
        '" y2="' +
        Y(c[id][1]).toFixed(1) +
        '" stroke="#98989D" stroke-width=".7" opacity=".45"/>';
  });
  Object.keys(c).forEach((id) => {
    const p = by[id];
    if (!p) return;
    s +=
      '<circle cx="' +
      X(c[id][0]).toFixed(1) +
      '" cy="' +
      Y(c[id][1]).toFixed(1) +
      '" r="3.6" fill="' +
      ic(p[2]) +
      '" stroke="#000" stroke-width=".8" data-id="' +
      id +
      '"/>';
  });
  let bi = null;
  Object.keys(c).forEach((id) => {
    const p = by[id];
    if (p && p[4] != null && (!bi || p[4] > by[bi][4])) bi = id;
  });
  if (bi) {
    const bx = X(c[bi][0]),
      byy = Y(c[bi][1]),
      lft = bx < L + (W - L) / 2;
    s +=
      '<circle cx="' +
      bx.toFixed(1) +
      '" cy="' +
      byy.toFixed(1) +
      '" r="9" fill="none" stroke="#F5F5F7" stroke-width="1.2"/><text x="' +
      (bx + (lft ? 15 : -15)).toFixed(1) +
      '" y="' +
      (byy + 4).toFixed(1) +
      '" fill="#F5F5F7" font-size="13" font-weight="600" text-anchor="' +
      (lft ? "start" : "end") +
      '">best ' +
      f4(by[bi][4]) +
      "</text>";
  }
  s +=
    '<text x="' +
    (L + (W - L) / 2) +
    '" y="' +
    (H - 10) +
    '" fill="#98989D" font-size="12" text-anchor="middle">code, component 1</text><text x="16" y="' +
    (T + (H - T - B) / 2) +
    '" fill="#98989D" font-size="12" text-anchor="middle" transform="rotate(-90 16 ' +
    (T + (H - T - B) / 2) +
    ')">code, component 2</text>';
  return s + "</svg>";
}
el("heroDisk").innerHTML = diskSvg(R[NAMES[0]], 640);
chordLabel(el("heroDisk"));
hover(el("heroDisk"), R[NAMES[0]]);
el("heroCode").innerHTML = codeSvg(R[NAMES[0]], 800, 560);
hover(el("heroCode"), R[NAMES[0]]);
(() => {
  const hero = el("hero");
  const r = R[NAMES[0]];
  const by = byId(r);
  let last = null;
  function lineage(id) {
    const set = {};
    let k = 0;
    while (id && by[id] && !set[id] && k++ < 400) {
      set[id] = 1;
      id = by[id][1];
    }
    return set;
  }
  hero.addEventListener("mousemove", (e) => {
    const id = e.target.getAttribute && e.target.getAttribute("data-id");
    if (id === last) return;
    last = id;
    hero.querySelectorAll(".hl").forEach((n) => {
      n.classList.remove("hl");
    });
    if (!id) {
      hero.classList.remove("brush");
      return;
    }
    const set = lineage(id);
    hero.classList.add("brush");
    hero.querySelectorAll("[data-id],[data-l]").forEach((n) => {
      const k = n.getAttribute("data-id") || n.getAttribute("data-l");
      if (set[k]) n.classList.add("hl");
    });
  });
  hero.addEventListener("mouseleave", () => {
    last = null;
    hero.classList.remove("brush");
    hero.querySelectorAll(".hl").forEach((n) => {
      n.classList.remove("hl");
    });
  });
})();
