"""Hexagon packing: 11 unit hexagons in a regular hexagon, minimize outer side."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 118.0  # safety margin within 150s limit


def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts


def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """For fixed outer hex angle, find optimal center and side via LP.
    Returns (cx, cy, side, angle_deg)."""
    all_verts = []
    for ix, iy, iang in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, iang))

    n_pts = len(all_verts)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg

    A_ub = []
    b_ub = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_verts:
            # (vx - cx)*nx + (vy - cy)*ny <= t  =>  -nx*cx - ny*cy - t <= -nx*vx - ny*vy
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            # -(vx - cx)*nx - (vy - cy)*ny <= t  =>  nx*cx + ny*cy - t <= nx*vx + ny*vy
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(None, None), (None, None), (0, None)],
                         method='highs')
        if result.success:
            cx, cy, t = result.x[0], result.x[1], result.x[2]
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side, angle_deg
    except Exception:
        pass

    # Fallback: centroid + max projection
    cx = sum(v[0] for v in all_verts) / n_pts
    cy = sum(v[1] for v in all_verts) / n_pts
    t = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        for vx, vy in all_verts:
            proj = (vx - cx) * nx + (vy - cy) * ny
            t = max(t, proj)
    side = t / (math.sqrt(3) / 2)
    return cx, cy, side, angle_deg


def _min_enclosing_hex(inner_hexes):
    """Find minimum enclosing regular hexagon over all angles via multi-resolution sweep."""
    best_side = float('inf')
    best_cx, best_cy, best_ang = 0.0, 0.0, 0.0

    # Coarse sweep: 0 to 60 degrees (periodicity), step 2
    for ang in np.arange(0, 60, 2.0):
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)

    # Fine sweep around best coarse
    for ang in np.arange(max(0, best_ang - 4), min(60, best_ang + 4), 0.25):
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)

    # Finer sweep
    for ang in np.arange(max(0, best_ang - 1), min(60, best_ang + 1), 0.05):
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)

    # Ultra-fine polish
    for ang in np.arange(max(0, best_ang - 0.2), min(60, best_ang + 0.2), 0.005):
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)

    return best_cx, best_cy, best_side, best_ang


def _hexagonal_lattice_placement(n_hex, gap_factor=1.001):
    """Place hexagons in a compact hexagonal lattice pattern.
    gap_factor=1.001 gives minimal separation (0.1% of center distance) ensuring disjointness."""
    d = math.sqrt(3) * gap_factor
    u = np.array([d, 0.0])
    v = np.array([d * 0.5, d * math.sqrt(3) / 2])

    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            pos = i * u + j * v
            dist_sq = pos[0] ** 2 + pos[1] ** 2
            candidates.append((dist_sq, pos[0], pos[1]))

    candidates.sort(key=lambda t: t[0])

    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == n_hex:
            break

    # All hexagons at 0 degrees (flat-top orientation aligns with lattice)
    return [[x, y, 0.0] for x, y in positions]


def _try_inner_angle_variants(inner_hexes, time_budget=15.0):
    """Try rotating all inner hexagons to see if a different orientation helps."""
    best_inner = inner_hexes
    best_side = float('inf')
    best_outer = (0.0, 0.0, 0.0, 0.0)

    cx, cy, side, ang = _min_enclosing_hex(inner_hexes)
    best_side = side
    best_outer = (cx, cy, side, ang)

    t_start = time.time()
    # Try a few rotation offsets
    for rot in [0.0, 15.0, 30.0, 45.0]:
        if time.time() - t_start > time_budget:
            break
        rotated = [[x, y, 0.0 + rot] for x, y, _ in inner_hexes]
        cx2, cy2, side2, ang2 = _min_enclosing_hex(rotated)
        if side2 < best_side:
            best_side = side2
            best_inner = rotated
            best_outer = (cx2, cy2, side2, ang2)

    return best_inner, best_outer


def _polish_outer(inner_hexes, cx0, cy0, ang0, side0, time_budget=10.0):
    """Lightweight polish: use smooth surrogate to refine outer hex center and angle.
    Only adjusts (cx, cy, angle) — does NOT move inner hexagons."""
    t_start = time.time()

    all_verts = []
    for ix, iy, iang in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, iang))

    def outer_objective(params):
        cx, cy, ang = params[0], params[1], math.degrees(params[2])
        if time.time() - t_start > time_budget:
            return side0
        # Compute max projection over 6 face normals
        t = 0.0
        for k in range(6):
            a = math.radians(ang + 30 + 60 * k)
            nx, ny = math.cos(a), math.sin(a)
            for vx, vy in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                t = max(t, proj)
        return t / (math.sqrt(3) / 2)

    x0 = np.array([cx0, cy0, math.radians(ang0)])
    try:
        result = minimize(outer_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
        if result.fun < side0 * (1 - 1e-9):
            cx, cy, ang = result.x[0], result.x[1], math.degrees(result.x[2])
            return cx, cy, result.fun, ang
    except Exception:
        pass

    return cx0, cy0, side0, ang0


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = _hexagonal_lattice_placement(N)
        cx, cy, side, ang = _min_enclosing_hex(inner)
        return inner, (cx, cy), side, ang

    # Step 1: Compact lattice placement with minimal gap
    inner = _hexagonal_lattice_placement(N, gap_factor=1.001)

    # Step 2: Thorough outer-angle search
    if time.time() < _DEADLINE - 30.0:
        cx, cy, side, ang = _min_enclosing_hex(inner)
    else:
        cx, cy, side, ang = _min_enclosing_hex_lp(inner, 0.0)

    # Step 3: Try inner angle variants (lightweight)
    if time.time() < _DEADLINE - 20.0:
        inner2, outer2 = _try_inner_angle_variants(inner, time_budget=12.0)
        if outer2[2] < side:
            inner, (cx, cy, side, ang) = inner2, outer2

    # Step 4: Polish outer hex parameters
    if time.time() < _DEADLINE - 10.0:
        cx, cy, side, ang = _polish_outer(inner, cx, cy, ang, side, time_budget=8.0)

    # Final safety check: ensure we have a valid result
    if side <= 0 or math.isnan(side) or math.isinf(side):
        inner = _hexagonal_lattice_placement(N, gap_factor=1.001)
        cx, cy, side, ang = _min_enclosing_hex_lp(inner, 0.0)

    return inner, (cx, cy), side, ang
