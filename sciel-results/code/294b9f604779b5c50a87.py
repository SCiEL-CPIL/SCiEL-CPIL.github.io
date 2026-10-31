"""Fresh design: rigid 11-hex cluster + global/outer optimization with SAT checks."""
import math
import time
import numpy as np

N = 11
DEADLINE = 138.0  # safety margin vs 150s kill
_T0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _T0)


def _hex_verts(cx, cy, s, ang):
    a = math.radians(ang)
    c, sn = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        px, py = s * math.cos(th), s * math.sin(th)
        out.append((cx + px * c - py * sn, cy + px * sn + py * c))
    return out


def _sep(p, a, b):
    # signed distance from p to oriented edge a->b (positive = left)
    ax, ay = a; bx, by = b; px, py = p
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * ex + (py - ay) * ey) / L2
    if t <= 0.0:
        return math.hypot(px - ax, py - ay)
    if t >= 1.0:
        return math.hypot(px - bx, py - by)
    # perpendicular distance
    d = (ex * (py - ay) - ey * (px - ax)) / math.sqrt(L2)
    return d


def _poly_sep(A, B):
    """Max penetration depth (positive => separated by that amount, negative => overlap)."""
    best = -1e18
    nA, nB = len(A), len(B)
    for i in range(nA):
        a1 = A[i]; a2 = A[(i + 1) % nA]
        ex, ey = a2[0] - a1[0], a2[1] - a1[1]
        L = math.hypot(ex, ey)
        if L == 0:
            continue
        nx, ny = -ey / L, ex / L  # outward normal (CCW polygon)
        dmax = -1e18
        for p in B:
            d = nx * (p[0] - a1[0]) + ny * (p[1] - a1[1])
            if d > dmax:
                dmax = d
        if dmax > best:
            best = dmax
    for i in range(nB):
        b1 = B[i]; b2 = B[(i + 1) % nB]
        ex, ey = b2[0] - b1[0], b2[1] - b1[1]
        L = math.hypot(ex, ey)
        if L == 0:
            continue
        nx, ny = -ey / L, ex / L
        dmax = -1e18
        for p in A:
            d = nx * (p[0] - b1[0]) + ny * (p[1] - b1[1])
            if d > dmax:
                dmax = d
        if dmax > best:
            best = dmax
    return best


def _cluster_verts(gx, gy, grot_deg, rel):
    """Return list of 11 hexagon vertex-lists for a cluster config."""
    gr = math.radians(grot_deg)
    cg, sg = math.cos(gr), math.sin(gr)
    verts = []
    for (rx, ry, ra) in rel:
        # rotate relative position by global rotation
        wx = gx + rx * cg - ry * sg
        wy = gy + rx * sg + ry * cg
        ang = ra + grot_deg
        verts.append(_hex_verts(wx, wy, 1.0, ang))
    return verts


def _build_rel():
    """Tight rigid cluster: center + 6 neighbors + 5 more in a 12-cell hex region minus 1."""
    # axial coords -> cartesian (pointy? we use flat-top with 60deg steps)
    # Use cube/axial: x = 3/2 * q, y = sqrt(3)/2 * q + sqrt(3) * r  (for side 1, center-to-center = sqrt(3))
    s3 = math.sqrt(3.0)
    cells = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q + r) <= 1:
                cells.append((q, r))
    # cells count = 7 (hex of radius 1). Need 11 -> expand to radius 2 region but keep tight.
    # Build radius-2 hex (19 cells), drop 8 outer to make 11 while staying compact.
    cells2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                cells2.append((q, r))
    # choose 11: center + ring1 (6) + 5 of ring2
    center = (0, 0)
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2 = []
    for (q, r) in cells2:
        if (q, r) != center and (q, r) not in ring1:
            ring2.append((q, r))
    # pick 5 ring2 cells that keep cluster compact (prefer those adjacent to ring1)
    picked = [center] + ring1
    for (q, r) in ring2:
        if len(picked) >= 11:
            break
        # check adjacency to existing
        adj = any(abs(q - a) + abs(r - b) + abs((q + r) - (a + b)) <= 2 for (a, b) in picked)
        if adj:
            picked.append((q, r))
    # fallback fill if < 11
    if len(picked) < 11:
        for (q, r) in ring2:
            if (q, r) not in picked:
                picked.append((q, r))
            if len(picked) >= 11:
                break
    picked = picked[:11]
    rel = []
    for (q, r) in picked:
        x = 1.5 * q * 1.0  # side 1 -> center distance factor
        y = s3 / 2.0 * q + s3 * r
        rel.append((x, y, 0.0))
    return rel


REL = _build_rel()


def _required_side(verts, ocx, ocy, oside, orot_deg):
    """Return minimal oside such that all verts fit in outer hex (center oc, rot orot)."""
    or_ = math.radians(orot_deg)
    co, so = math.cos(or_), math.sin(or_)
    # outer hex vertices (side s, flat-top, centered at oc)
    # we need max over outer edges of the support of verts; required side = max_depth / (cos(30deg) factor)
    # For a regular hexagon of side s, the distance from center to each edge (inradius) = s * sqrt(3)/2.
    # A point fits iff for every outer edge, the point is on the interior side by at least 0.
    # required s = max over edges of (distance from center to the supporting line through the extreme vert).
    # Compute extreme projection of verts onto each outer edge normal.
    best = 0.0
    for i in range(6):
        th = or_ + i * math.pi / 3.0
        nx, ny = math.cos(th), math.sin(th)  # outward normal
        # extreme vert projection relative to center
        dmax = -1e18
        for v in verts:
            d = nx * (v[0] - ocx) + ny * (v[1] - ocy)
            if d > dmax:
                dmax = d
        # inradius needed = dmax (distance from center to supporting line)
        # side = inradius / (sqrt(3)/2)
        s_need = dmax / (math.sqrt(3.0) / 2.0)
        if s_need > best:
            best = s_need
    return max(best, 0.0)


def _cluster_disjoint(verts):
    """Check all 11 hexagons pairwise disjoint via SAT."""
    n = len(verts)
    for i in range(n):
        for j in range(i + 1, n):
            if _poly_sep(verts[i], verts[j]) < -1e-9:
                return False
    return True


def _score_config(gx, gy, grot_deg, ocx, ocy, orot_deg):
    verts = _cluster_verts(gx, gy, grot_deg, REL)
    if not _cluster_disjoint(verts):
        return None
    oside = _required_side(verts, ocx, ocy, oside=1.0, orot_deg=orot_deg)
    # verify fit
    return oside


def construct():
    global _T0
    _T0 = time.time()

    best = (999.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # side, gx, gy, grot, orot, ocenter
    best_side = 999.0
    best_verts = None

    # Phase 1: coarse grid over cluster rotation & outer rotation, center cluster at origin, outer center = cluster centroid
    # Compute cluster centroid
    cx = sum(r[0] for r in REL) / len(REL)
    cy = sum(r[1] for r in REL) / len(REL)

    def try_cfg(gx, gy, grot, orot):
        nonlocal best_side, best, best_verts
        if _time_left() < 1.0:
            return
        verts = _cluster_verts(gx, gy, grot, REL)
        if not _cluster_disjoint(verts):
            return
        # outer center = centroid of cluster verts
        allpts = [p for v in verts for p in v]
        ocx = sum(p[0] for p in allpts) / len(allpts)
        ocy = sum(p[1] for p in allpts) / len(allpts)
        oside = _required_side(verts, ocx, ocy, 1.0, orot)
        if oside < best_side:
            best_side = oside
            best = (oside, gx, gy, grot, orot, (ocx, ocy))
            best_verts = verts

    # coarse
    for grot in range(0, 60, 5):
        for orot in range(0, 60, 5):
            try_cfg(-cx, -cy, float(grot), float(orot))

    # Phase 2: local refinement around best (cluster rot, outer rot)
    if best_side < 999.0:
        s0, gx0, gy0, grot0, orot0, _ = best
        for _ in range(3):
            for dg in [-3, -1, 1, 3]:
                for do in [-3, -1, 1, 3]:
                    try_cfg(gx0, gy0, grot0 + dg, orot0 + do)

    # Phase 3: optimize outer center (2D) and cluster translation to minimize side, given rotations
    def refine_center(gx, gy, grot, orot, iters=30):
        nonlocal best_side, best, best_verts
        verts = _cluster_verts(gx, gy, grot, REL)
        if not _cluster_disjoint(verts):
            return
        allpts = [p for v in verts for p in v]
        ocx = sum(p[0] for p in allpts) / len(allpts)
        ocy = sum(p[1] for p in allpts) / len(allpts)
        oside = _required_side(verts, ocx, ocy, 1.0, orot)
        step = max(oside * 0.05, 0.05)
        for _ in range(iters):
            if _time_left() < 1.0:
                break
            improved = False
            for dx in (-step, 0, step):
                for dy in (-step, 0, step):
                    cand = _required_side(verts, ocx + dx, ocy + dy, 1.0, orot)
                    if cand < oside - 1e-12:
                        oside = cand
                        ocx += dx
                        ocy += dy
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
        if oside < best_side:
            best_side = oside
            best = (oside, gx, gy, grot, orot, (ocx, ocy))
            best_verts = verts

    # refine center for a few best rotation combos
    if best_side < 999.0:
        s0, gx0, gy0, grot0, orot0, _ = best
        refine_center(gx0, gy0, grot0, orot0, iters=40)

    # Phase 4: fine rotation sweep near best with center re-optimization
    if best_side < 999.0 and _time_left() > 5.0:
        s0, gx0, gy0, grot0, orot0, _ = best
        for dg in (-2, -1, 0, 1, 2):
            for do in (-2, -1, 0, 1, 2):
                if _time_left() < 2.0:
                    break
                refine_center(gx0, gy0, grot0 + dg, orot0 + do, iters=25)

    # Final: build inner list from best config
    if best_side >= 999.0:
        # fallback: trivial valid config (well separated)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    s0, gx0, gy0, grot0, orot0, (ocx, ocy) = best
    inner = []
    for (rx, ry, ra) in REL:
        gr = math.radians(grot0)
        cg, sg = math.cos(gr), math.sin(gr)
        wx = gx0 + rx * cg - ry * sg
        wy = gy0 + rx * sg + ry * cg
        ang = ra + grot0
        inner.append([wx, wy, ang])
    return inner, (ocx, ocy), float(best_side), float(orot0)


# ensure module-level call safety (construct is the entry point)
