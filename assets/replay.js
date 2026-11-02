let tmax = null,
  playing = null;
function diskDraw() {
  const r = R[cur],
    f = el("disk");
  f.innerHTML = diskSvg(r, 720, tmax);
  chordLabel(f);
  hover(f, r);
  const mx = Math.max.apply(
      null,
      r.programs.map((p) => p[6] || 0),
    ),
    shownN = r.programs.filter((p) => tmax == null || (p[6] || 0) <= tmax).length;
  el("tlabel").textContent = (tmax == null ? "attempt " + mx : "attempt " + tmax) + " · " + shownN + " programs";
}
function disk() {
  const r = R[cur],
    mx = Math.max.apply(
      null,
      r.programs.map((p) => p[6] || 0),
    );
  el("tslider").max = mx;
  el("tslider").value = mx;
  tmax = null;
  diskDraw();
  if (diskLoop) dplay();
  const rose = r.transfers.filter((t) => t.rose).length,
    isl = {};
  r.programs.forEach((p) => {
    isl[p[2]] = 1;
  });
  el("s-disk").innerHTML = r.cells.length
    ? "In " +
      LBL(cur) +
      " the " +
      Object.keys(isl).length +
      " demes grew " +
      r.programs.length +
      " programs. " +
      r.cells.length +
      " research cells wrote " +
      r.nfind +
      " findings, delivered " +
      r.delivered +
      " times. " +
      r.transfers.length +
      " reached another deme as a chord, and " +
      rose +
      " of those raised the child there above its parent."
    : "In " +
      LBL(cur) +
      " research cells were off: the demes grew " +
      r.programs.length +
      " programs as separate trees and nothing crossed between them. Compare with an autocorrelation run, where chords appear.";
  el("findings").innerHTML = r.findings.length
    ? r.findings
        .slice(0, 5)
        .map(
          (x) =>
            '<div class="row">' +
            esc((x.text || "").length > 230 ? x.text.slice(0, 230) + "…" : x.text) +
            '<div class="m">deme ' +
            x.island +
            " · delivered " +
            (x.delivered || 0) +
            " · " +
            esc(x.status) +
            " · " +
            esc(x.kind) +
            "</div></div>",
        )
        .join("")
    : '<div class="row muted">No research cells in this run.</div>';
  el("diskleg").innerHTML =
    Object.keys(isl)
      .sort((a, b) => a - b)
      .map((k) => '<span><i style="background:' + ic(+k) + '"></i>deme ' + k + "</span>")
      .join("") +
    '<span><i class="ln" style="background:var(--chord)"></i>finding, child improved</span><span><i class="dash"></i>finding, child did not</span>';
}
var diskLoop = false;
let dhold = null;
function dstop() {
  clearInterval(playing);
  playing = null;
  clearTimeout(dhold);
}
function dplay() {
  dstop();
  diskLoop = true;
  el("play").textContent = "Pause";
  const mx = +el("tslider").max;
  tmax = 0;
  playing = setInterval(() => {
    tmax = Math.min(mx, tmax + Math.max(1, Math.round(mx / 160)));
    el("tslider").value = tmax;
    diskDraw();
    if (tmax >= mx) {
      clearInterval(playing);
      playing = null;
      dhold = setTimeout(() => {
        if (diskLoop) dplay();
      }, 2600);
    }
  }, 70);
}
el("tslider").addEventListener("input", function () {
  diskLoop = false;
  dstop();
  el("play").textContent = "Play";
  tmax = +this.value;
  diskDraw();
});
el("play").addEventListener("click", function () {
  if (diskLoop) {
    diskLoop = false;
    dstop();
    this.textContent = "Play";
  } else dplay();
});
new IntersectionObserver(
  (es) => {
    es.forEach((en) => {
      if (en.isIntersecting && !diskLoop && !el("disk").dataset.on) {
        el("disk").dataset.on = 1;
        dplay();
      }
    });
  },
  { threshold: 0.3 },
).observe(el("disk"));
listeners.push(disk);
disk();
