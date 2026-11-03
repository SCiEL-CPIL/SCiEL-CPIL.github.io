"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Optimal rigid cluster (center + 6 ring-1 + 4 ring-2) with dense sweep over
outer hexagon orientation. LP gives exact min side for each orientation.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
DEADLINE = 150.0
SAFETY = 10.0
SQRT3 = math.sqrt(3.0)


def _hex_vertices(cx, cy, side, angle_deg):
    a0 = math.radians(angle_deg)
    return [(cx + side * math.cos(a0 + math.radians(60 * k)),
             cy + side * math.sin(a0 + math.radians(60 * k))) for k in range(6)]


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2.0
    for k in range(6):
        an = math.radians(30 + 60 * k)
        if lx * math.cos(an) + ly * math.sin(an) > apothem + 1e-8:
            return False
    return True


def _segs_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hex_disjoint(h1, h2, eps=1e-9):
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    dx = cx2 - cx1; dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    v1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in v1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    v2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in v2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        for j in range(6):
            if _segs_intersect(v1[i], v1[(i + 1) % 6], v2[j], v2[(j + 1) % 6]):
                return False
    return True


def _all_disjoint(hexes):
    n = len(hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hex_disjoint(hexes[i], hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    cx, cy, s, a = inner_hex
    ox, oy, oside, oa = outer_hex
    for vx, vy in _hex_vertices(cx, cy, s, a):
        if not _point_in_hex(vx, vy, ox, oy, oside, oa):
            return False
    return True


def _axial_to_xy(q, r):
    return (1.5 * q, SQRT3 * (r + q / 2.0))


def _build_optimal_cluster():
    """Center + 6 ring-1 + 4 ring-2 in the known optimal configuration."""
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    # 4 ring-2 hexagons: pick symmetric ones for compactness
    # (2,-1), (0,-2), (-2,-1), (-1,2) or similar - try a few
    ring2_options = [
        [(2, -1), (0, -2), (-2, -1), (-1, 2)],
        [(2, 0), (0, -2), (-2, 0), (0, 2)],
        [(2, -1), (1, -2), (-1, -2), (-2, -1)],
        [(2, 0), (1, -1), (-1, -1), (-2, 0)],
        [(2, -1), (0, -2), (-2, -1), (-2, 1)],
        [(1, -2), (0, -2), (-1, -2), (0, 2)],
    ]
    return ring1, ring2_options


def _outer_side_for_rigid(pts, oangle_deg):
    """Min side of outer hexagon (orientation oangle) enclosing all vertices of
    unit hexagons centered at pts (all inner angle 0)."""
    allv = []
    for (px, py) in pts:
        for k in range(6):
            a = math.radians(60 * k)
            allv.append((px + math.cos(a), py + math.sin(a)))
    allv = np.array(allv)

    base = np.radians(30 + 60 * np.arange(6))
    nxs = np.cos(base); nys = np.sin(base)

    best_t = 0.0
    for k in range(6):
        proj = allv[:, 0] * nxs[k] + allv[:, 1] * nys[k]
        mx = proj.max()
        if mx > best_t:
            best_t = mx

    A = []
    b = []
    for k in range(6):
        A.append([nxs[k], nys[k], -1.0]); b.append(0.0)
        A.append([-nxs[k], -nys[k], -1.0]); b.append(0.0)
    A.append([0.0, 0.0, -1.0]); b.append(-best_t)
    A = np.array(A); b = np.array(b)

    res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A, b_ub=b,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if not res.success:
        return (float('inf'), 0.0, 0.0)
    t = res.x[2]
    side = t * 2.0 / SQRT3
    return (side, res.x[0], res.x[1])


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY

    ring1, ring2_options = _build_optimal_cluster()
    
    # Build all candidate clusters
    all_clusters = []
    base_pts = [(0.0, 0.0)]
    for q, r in ring1:
        base_pts.append(_axial_to_xy(q, r))
    
    for r2 in ring2_options:
        pts = list(base_pts)
        for q, r in r2:
            pts.append(_axial_to_xy(q, r))
        all_clusters.append(pts)
    
    # Also try all C(14,4) combinations of ring-2 for thoroughness
    # But first do a quick pass on the curated options
    best_side = float('inf')
    best_pts = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Dense sweep over [0, 60) for curated clusters
    for pts in all_clusters:
        if time.time() > deadline:
            break
        # Sweep in 0.5 degree steps
        a = 0.0
        while a < 60.0 + 1e-9:
            s, cx, cy = _outer_side_for_rigid(pts, a)
            if s < best_side:
                best_side = s
                best_pts = pts
                best_center = (cx, cy)
                best_oangle = a
            a += 0.5

    # Phase 2: Fine sweep around best
    if best_pts is not None and time.time() < deadline - 2.0:
        lo = max(0.0, best_oangle - 1.0)
        hi = min(60.0, best_oangle + 1.0)
        a = lo
        while a <= hi + 1e-9:
            s, cx, cy = _outer_side_for_rigid(best_pts, a)
            if s < best_side:
                best_side = s
                best_center = (cx, cy)
                best_oangle = a
            a += 0.05

    # Phase 3: Ultra-fine
    if best_pts is not None and time.time() < deadline - 1.0:
        lo = max(0.0, best_oangle - 0.1)
        hi = min(60.0, best_oangle + 0.1)
        a = lo
        while a <= hi + 1e-9:
            s, cx, cy = _outer_side_for_rigid(best_pts, a)
            if s < best_side:
                best_side = s
                best_center = (cx, cy)
                best_oangle = a
            a += 0.005

    # If still have time, try more ring-2 combinations
    if time.time() < deadline - 5.0:
        import itertools
        ring2_all = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                     (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                     (0, 2), (1, 2)]
        # Skip ones already tried
        tried = set(tuple(sorted(r)) for r in ring2_options)
        for comb in itertools.combinations(range(14), 4):
            if time.time() > deadline - 2.0:
                break
            r2 = [ring2_all[i] for i in comb]
            key = tuple(sorted(r2))
            if key in tried:
                continue
            tried.add(key)
            pts = list(base_pts)
            for q, r in r2:
                pts.append(_axial_to_xy(q, r))
            # Quick coarse sweep
            a = 0.0
            while a < 60.0 + 1e-9:
                s, cx, cy = _outer_side_for_rigid(pts, a)
                if s < best_side:
                    best_side = s
                    best_pts = pts
                    best_center = (cx, cy)
                    best_oangle = a
                a += 2.0

        # Fine sweep on any improved cluster
        if time.time() < deadline - 1.0 and best_pts is not None:
            lo = max(0.0, best_oangle - 1.0)
            hi = min(60.0, best_oangle + 1.0)
            a = lo
            while a <= hi + 1e-9:
                s, cx, cy = _outer_side_for_rigid(best_pts, a)
                if s < best_side:
                    best_side = s
                    best_center = (cx, cy)
                    best_oangle = a
                a += 0.1

    # Fallback
    if best_pts is None:
        best_pts = list(base_pts) + [_axial_to_xy(2, -1), _axial_to_xy(0, -2),
                                     _axial_to_xy(-2, -1), _axial_to_xy(-1, 2)]
        s, cx, cy = _outer_side_for_rigid(best_pts, 0.0)
        best_side = s
        best_center = (cx, cy)
        best_oangle = 0.0

    # Build inner hexagons
    inner_hexes = [(best_pts[i][0], best_pts[i][1], 1.0, 0.0) for i in range(len(best_pts))]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02, 1.03]:
            scaled = [(x * scale, y * scale, 1.0, 0.0) for (x, y, _, _) in inner_hexes]
            if _all_disjoint(scaled):
                inner_hexes = scaled
                best_pts = [(h[0], h[1]) for h in inner_hexes]
                s, cx, cy = _outer_side_for_rigid(best_pts, best_oangle)
                best_side = s
                best_center = (cx, cy)
                break

    # Verify containment
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    for _ in range(200):
        if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
            break
        best_side *= 1.0005
        outer_hex = (best_center[0], best_center[1], best_side, best_oangle)

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(len(inner_hexes))]
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
