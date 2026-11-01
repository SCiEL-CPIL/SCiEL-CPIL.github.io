"""Hexagon packing: smooth surrogate + Nelder-Mead from lattice seeds."""
import time
import math
import numpy as np

N = 11

# Precompute unit regular hexagon vertices (flat-top, circumradius = 1)
_HEX_VERTS = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 7)[:-1]])


def _hex_verts(cx, cy, s, ang):
    """Vertices of a regular hexagon, side s, center (cx,cy), rotation ang (radians)."""
    c, si = math.cos(ang), math.sin(ang)
    R = np.array([[c, -si], [si, c]])
    return s * (_HEX_VERTS @ R.T) + np.array([cx, cy])


def _point_in_hex(v, cx, cy, s, ang):
    """Signed-ish: True if vertex v is strictly inside hexagon (center, side, angle)."""
    c, si = math.cos(ang), math.sin(ang)
    # transform point into hexagon's local frame
    px = (v[0] - cx) * c + (v[1] - cy) * si
    py = -(v[0] - cx) * si + (v[1] - cy) * c
    ap = 3.0 * s / 2.0  # apothem
    # regular hexagon: |x| <= ap and |y| <= ap and |x|+|y|*sqrt(3) <= 2*ap ... use edge form
    # edges normals at 0,60,120,180,240,300 deg
    ok = True
    for k in range(6):
        a = math.radians(60 * k)
        nx, ny = math.cos(a), math.sin(a)
        if nx * px + ny * py > ap + 1e-12:
            ok = False
            break
    return ok


def _dist_to_outer(v, cx, cy, s, ang):
    """Distance from vertex v to the outer hexagon boundary (negative if inside)."""
    c, si = math.cos(ang), math.sin(ang)
    px = (v[0] - cx) * c + (v[1] - cy) * si
    py = -(v[0] - cx) * si + (v[1] - cy) * c
    ap = 3.0 * s / 2.0
    d = 0.0
    for k in range(6):
        a = math.radians(60 * k)
        nx, ny = math.cos(a), math.sin(a)
        d = max(d, nx * px + ny * py - ap)
    return d


def _smooth_dist_to_outer(v, cx, cy, s, ang, t):
    c, si = math.cos(ang), math.sin(ang)
    px = (v[0] - cx) * c + (v[1] - cy) * si
    py = -(v[0] - cx) * si + (v[1] - cy) * c
    ap = 3.0 * s / 2.0
    vals = []
    for k in range(6):
        a = math.radians(60 * k)
        nx, ny = math.cos(a), math.sin(a)
        vals.append(nx * px + ny * py - ap)
    m = max(vals)
    sm = (m + t * math.log(math.exp((m - m) / t))) if t > 0 else m  # log-sum-exp approx (m dominates)
    # log-sum-exp of vals
    if t > 1e-9:
        sm = t * math.log(sum(math.exp((x - m) / t) for x in vals)) + m
    else:
        sm = m
    return sm


def _smooth_minimax(params, t, oang):
    """params: [x1,y1,a1, x2,y2,a2, ...]. Returns smooth max of vertex distances to outer boundary."""
    s_outer = None  # not needed here; outer fixed during optimization
    # outer boundary params are global-ish; pass via closure instead
    return _smooth_minimax_with(params, t, oang, _OUTER)


def _smooth_minimax_with(params, t, oang, outer):
    ocx, ocy, os = outer
    worst = -1e9
    n = len(params) // 3
    for i in range(n):
        x, y, a = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts = _hex_verts(x, y, 1.0, a)
        for v in verts:
            d = _smooth_dist_to_outer(v, ocx, ocy, os, oang, t)
            if d > worst:
                worst = d
    return worst


def _feasible(params, outer, tol=1e-6):
    ocx, ocy, os, oang = outer
    for i in range(len(params) // 3):
        x, y, a = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts = _hex_verts(x, y, 1.0, a)
        for v in verts:
            if not _point_in_hex(v, ocx, ocy, os, oang):
                return False
    return True


def _outer_side_for_config(params, ocx, ocy, oang, lo, hi, iters=40):
    """Smallest outer side s such that all inner hexes fit in outer hex (center fixed, angle fixed)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _feasible(params, (ocx, ocy, mid, oang)):
            hi = mid
        else:
            lo = mid
    return hi


def _lattice_seed():
    """Hexagonal lattice of N unit hexagons (side 1), centers on triangular grid spacing 2 (touching)."""
    # For unit side hexes, center-to-center touching distance = 2 * apothem = 3.
    # Use spacing 3.0 in two directions 60 deg apart.
    d = 3.0
    a1 = np.array([d, 0.0])
    a2 = np.array([d * math.cos(math.radians(60)), d * math.sin(math.radians(60))])
    pts = []
    # generate indices in a hexagon-shaped region of size N
    # use axial coordinates with |q|,|r|,|q+r| bounded
    R = (N - 1) // 2
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            if abs(q + r) <= R:
                if len(pts) >= N:
                    break
                p = q * a1 + r * a2
                pts.append([float(p[0]), float(p[1]), 0.0])
        if len(pts) >= N:
            break
    # pad if short
    while len(pts) < N:
        pts.append([0.0, 0.0, 0.0])
    return pts[:N]


def _optimize_from_seed(seed_pts, outer, deadline):
    """Nelder-Mead on smooth surrogate to minimize required outer side (via boundary fit)."""
    from scipy.optimize import minimize
    params0 = np.array([c for p in seed_pts for c in p], dtype=float)
    ocx, ocy, os_init, oang = outer

    def obj(p):
        # smooth minimax of distances to outer boundary at current os_init
        return _smooth_minimax_with(p, 0.05, oang, (ocx, ocy, os_init, oang))

    # First, tighten outer to a feasible-but-tight box, then optimize positions to fit.
    # Strategy: start with generous outer, optimize to push boundary, then binary-search side.
    # Simpler: optimize positions to minimize smooth maxim over a fixed generous outer,
    # then binary search the true minimal side.
    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
    best_params = res.x
    # Now find minimal outer side via bisection with a few re-optimization passes
    s_lo = 0.0
    # upper bound: use current config's required side
    s_hi = _outer_side_for_config(best_params, ocx, ocy, oang, 0.0, 100.0, iters=50)
    for _ in range(3):
        # re-optimize positions for the mid side to tighten
        mid = 0.5 * (s_lo + s_hi)
        def obj2(p):
            return _smooth_minimax_with(p, 0.05, oang, (ocx, ocy, mid, oang))
        res2 = minimize(obj2, best_params, method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        best_params = res2.x
        if _feasible(best_params, (ocx, ocy, s_hi, oang)):
            s_hi = _outer_side_for_config(best_params, ocx, ocy, oang, 0.0, s_hi, iters=40)
        if time.time() > deadline:
            break
    final_side = _outer_side_for_config(best_params, ocx, ocy, oang, 0.0, s_hi, iters=50)
    inner = [[float(best_params[3 * i]), float(best_params[3 * i + 1]),
              math.degrees(best_params[3 * i + 2])] for i in range(len(best_params) // 3)]
    return inner, (ocx, ocy), final_side, math.degrees(oang)


def construct():
    global _OUTER
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s

    # Start with a generous outer box; we will shrink it.
    ocx, ocy = 0.0, 0.0
    oang = 0.0
    # generous initial side
    gen_side = 30.0
    _OUTER = (ocx, ocy, gen_side, oang)

    seed_pts = _lattice_seed()
    # initial feasibility: ensure seed fits in generous outer (it should)
    # find a tight-ish side for the seed as a starting point
    s_start = _outer_side_for_config(np.array([c for p in seed_pts for c in p]), ocx, ocy, oang, 0.0, gen_side, iters=40)

    best = None
    best_side = s_start
    # run optimization
    inner, center, side, angle = _optimize_from_seed(seed_pts, (ocx, ocy, s_start, oang), deadline)
    best_side = side
    best = (inner, center, side, angle)

    if time.time() < deadline:
        # second seed: slightly perturbed lattice for diversity
        seed2 = [ [p[0] + 0.0, p[1] + 0.0, math.radians(30)] for p in seed_pts ]
        s_start2 = _outer_side_for_config(np.array([c for p in seed2 for c in p]), ocx, ocy, oang, 0.0, gen_side, iters=40)
        inner2, center2, side2, angle2 = _optimize_from_seed(seed2, (ocx, ocy, s_start2, oang), deadline)
        if side2 < best_side:
            best = (inner2, center2, side2, angle2)
            best_side = side2

    return best


_OUTER = (0.0, 0.0, 30.0, 0.0)
