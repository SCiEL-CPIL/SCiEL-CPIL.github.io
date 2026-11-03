"""Packing 11 unit hexagons in a minimal regular hexagon using differentiable optimization."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of unit regular hexagon (side=1) centered at (cx,cy)."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _support_hex(origin_angle_rad, direction):
    """Support function of unit hexagon (circumradius=1) at origin in given direction."""
    d_angle = math.atan2(direction[1], direction[0])
    best = -float('inf')
    for k in range(6):
        va = origin_angle_rad + k * math.pi / 3
        c = math.cos(d_angle - va)
        if c > best:
            best = c
    return best

def _min_clearance_vec(p_new, a_new_deg, placed):
    """Min clearance between new hexagon and all placed. Positive = separated."""
    p_new = np.asarray(p_new, dtype=float)
    a_new_rad = math.radians(a_new_deg)
    min_m = float('inf')
    for (px, py, pa) in placed:
        diff = np.asarray([px, py]) - p_new
        a_placed_rad = math.radians(pa)
        # 12 critical directions
        for k in range(6):
            ang1 = a_new_rad + math.pi / 6 + k * math.pi / 3
            d = np.array([math.cos(ang1), math.sin(ang1)])
            s1 = _support_hex(a_new_rad, d)
            s2 = _support_hex(a_placed_rad, -d)
            m = s1 + s2 - np.dot(diff, d)
            if m < min_m:
                min_m = m
        for k in range(6):
            ang2 = a_placed_rad + math.pi / 6 + k * math.pi / 3
            d = np.array([math.cos(ang2), math.sin(ang2)])
            s1 = _support_hex(a_new_rad, d)
            s2 = _support_hex(a_placed_rad, -d)
            m = s1 + s2 - np.dot(diff, d)
            if m < min_m:
                min_m = m
    return min_m

def _check_overlap(p1, a1, p2, a2, eps=1e-8):
    """True if two unit hexagons overlap or touch."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    diff = p2 - p1
    a1r = math.radians(a1)
    a2r = math.radians(a2)
    min_m = float('inf')
    for k in range(6):
        ang1 = a1r + math.pi / 6 + k * math.pi / 3
        d = np.array([math.cos(ang1), math.sin(ang1)])
        s1 = _support_hex(a1r, d)
        s2 = _support_hex(a2r, -d)
        m = s1 + s2 - np.dot(diff, d)
        if m < min_m:
            min_m = m
    for k in range(6):
        ang2 = a2r + math.pi / 6 + k * math.pi / 3
        d = np.array([math.cos(ang2), math.sin(ang2)])
        s1 = _support_hex(a1r, d)
        s2 = _support_hex(a2r, -d)
        m = s1 + s2 - np.dot(diff, d)
        if m < min_m:
            min_m = m
    return min_m <= eps

def _enclosing_side_lp(inner_hexes, angle_deg):
    """LP to find min enclosing hexagon side for fixed outer angle."""
    all_verts = []
    for ix, iy, iang in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, iang))
    n_pts = len(all_verts)
    if n_pts == 0:
        return 0.0, 0.0, 0.0

    A_ub = []
    b_ub = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_verts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    if res.success:
        cx, cy, r = res.x[0], res.x[1], res.x[2]
    else:
        cx = sum(v[0] for v in all_verts) / n_pts
        cy = sum(v[1] for v in all_verts) / n_pts
        r = 0.0
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            nx, ny = math.cos(a), math.sin(a)
            for vx, vy in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                if proj > r:
                    r = proj
    side = r / (math.sqrt(3) / 2)
    return cx, cy, side

def _min_enclosing_side(inner_hexes):
    """Sweep outer angle to find minimal enclosing hexagon."""
    best_side = float('inf')
    best_cx = best_cy = 0.0
    best_ang = 0.0
    for ang in np.arange(0, 60, 2.0):
        if time.time() > _DEADLINE - 10:
            break
        cx, cy, s = _enclosing_side_lp(inner_hexes, float(ang))
        if s < best_side:
            best_side, best_cx, best_cy, best_ang = s, cx, cy, float(ang)
    if best_side < float('inf'):
        for ang in np.arange(max(0, best_ang - 5), min(60, best_ang + 5), 0.5):
            if time.time() > _DEADLINE - 10:
                break
            cx, cy, s = _enclosing_side_lp(inner_hexes, float(ang))
            if s < best_side:
                best_side, best_cx, best_cy, best_ang = s, cx, cy, float(ang)
    if best_side < float('inf'):
        for ang in np.arange(max(0, best_ang - 1), min(60, best_ang + 1), 0.1):
            if time.time() > _DEADLINE - 10:
                break
            cx, cy, s = _enclosing_side_lp(inner_hexes, float(ang))
            if s < best_side:
                best_side, best_cx, best_cy, best_ang = s, cx, cy, float(ang)
    return best_cx, best_cy, best_side, best_ang

def _lattice_placement():
    """Triangular lattice placement of 11 hexagons."""
    sqrt3 = math.sqrt(3)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            candidates.append((x * x + y * y, x, y))
    candidates.sort(key=lambda t: t[0])
    positions = []
    for _, x, y in candidates:
        positions.append((x, y, 30.0))
        if len(positions) == N:
            break
    return positions

def _greedy_placement():
    """Greedy max-min clearance placement."""
    sqrt3 = math.sqrt(3)
    placed = [(0.0, 0.0, 30.0)]
    for idx in range(1, N):
        if time.time() > _DEADLINE - 30:
            # fallback lattice
            u = (sqrt3 * 1.001, 0.0)
            v = (sqrt3 / 2 * 1.001, 1.5 * 1.001)
            cands = []
            for i in range(-5, 6):
                for j in range(-5, 6):
                    x = i * u[0] + j * v[0]
                    y = i * u[1] + j * v[1]
                    cands.append((x, y, 30.0))
            cands.sort(key=lambda c: c[0] ** 2 + c[1] ** 2)
            for c in cands:
                if not any(abs(c[0] - p[0]) < 0.01 and abs(c[1] - p[1]) < 0.01 for p in placed):
                    placed.append(c)
                    if len(placed) == N:
                        break
            return placed
        best_pos = None
        best_cl = -float('inf')
        search_r = 4.0
        step = 0.3
        for dx in np.arange(-search_r, search_r + step, step):
            for dy in np.arange(-search_r, search_r + step, step):
                if time.time() > _DEADLINE - 25:
                    break
                min_dist = min(math.sqrt((dx - px) ** 2 + (dy - py) ** 2) for px, py, _ in placed)
                if min_dist < sqrt3 * 0.95:
                    continue
                cl = _min_clearance_vec((dx, dy), 30.0, placed)
                if cl > best_cl:
                    best_cl = cl
                    best_pos = (dx, dy, 30.0)
        if best_pos is None:
            u = (sqrt3 * 1.001, 0.0)
            v = (sqrt3 / 2 * 1.001, 1.5 * 1.001)
            cands = []
            for i in range(-5, 6):
                for j in range(-5, 6):
                    x = i * u[0] + j * v[0]
                    y = i * u[1] + j * v[1]
                    cands.append((x, y, 30.0))
            cands.sort(key=lambda c: c[0] ** 2 + c[1] ** 2)
            for c in cands:
                if not any(abs(c[0] - p[0]) < 0.01 and abs(c[1] - p[1]) < 0.01 for p in placed):
                    placed.append(c)
                    if len(placed) == N:
                        break
            return placed
        placed.append(best_pos)
    return placed

def _repulsive_energy(params, n=N):
    """Energy: sum of overlap penalties + spread penalty.
    params: flat array [x0,y0,a0, x1,y1,a1, ...] angles in radians.
    """
    energy = 0.0
    for i in range(n):
        xi = params[3 * i]
        yi = params[3 * i + 1]
        ai = params[3 * i + 2]
        for j in range(i + 1, n):
            xj = params[3 * j]
            yj = params[3 * j + 1]
            aj = params[3 * j + 2]
            diff = np.array([xj - xi, yj - yi])
            # Compute min clearance using 12 directions
            min_m = float('inf')
            for k in range(6):
                ang1 = ai + math.pi / 6 + k * math.pi / 3
                d = np.array([math.cos(ang1), math.sin(ang1)])
                s1 = _support_hex(ai, d)
                s2 = _support_hex(aj, -d)
                m = s1 + s2 - np.dot(diff, d)
                if m < min_m:
                    min_m = m
            for k in range(6):
                ang2 = aj + math.pi / 6 + k * math.pi / 3
                d = np.array([math.cos(ang2), math.sin(ang2)])
                s1 = _support_hex(ai, d)
                s2 = _support_hex(aj, -d)
                m = s1 + s2 - np.dot(diff, d)
                if m < min_m:
                    min_m = m
            # Penalize overlap: if min_m < 0, penalty = (min_m)^2 * weight
            if min_m < 0:
                energy += 100.0 * min_m * min_m
    # Spread penalty: encourage packing within a reasonable radius
    cx = np.mean(params[0::3])
    cy = np.mean(params[1::3])
    for i in range(n):
        xi = params[3 * i] - cx
        yi = params[3 * i + 1] - cy
        energy += 0.01 * (xi * xi + yi * yi)
    return energy

def _nelder_mead_optimize(init_params, n=N, max_iter=500):
    """Nelder-Mead optimization of the packing."""
    def obj(p):
        return _repulsive_energy(p, n)
    res = minimize(obj, init_params, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    return res.x

def _params_to_hexes(params, n=N):
    """Convert flat params to list of (x, y, angle_deg)."""
    hexes = []
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        a_deg = math.degrees(params[3 * i + 2]) % 360
        hexes.append((x, y, a_deg))
    return hexes

def _validate_hexes(hexes, eps=1e-6):
    """Check all pairs are non-overlapping."""
    for i in range(len(hexes)):
        for j in range(i + 1, len(hexes)):
            if _check_overlap(hexes[i][:2], hexes[i][2], hexes[j][:2], hexes[j][2], eps=eps):
                return False
    return True

def _try_optimize_from_init(init_hexes, max_iter=300):
    """Optimize from initial hex positions using Nelder-Mead."""
    params = np.zeros(3 * N)
    for i, (x, y, a) in enumerate(init_hexes):
        params[3 * i] = x
        params[3 * i + 1] = y
        params[3 * i + 2] = math.radians(a)
    res = _nelder_mead_optimize(params, N, max_iter)
    hexes = _params_to_hexes(res, N)
    # Normalize: shift so centroid is at origin
    cx = sum(h[0] for h in hexes) / N
    cy = sum(h[1] for h in hexes) / N
    hexes = [(h[0] - cx, h[1] - cy, h[2]) for h in hexes]
    if _validate_hexes(hexes):
        return hexes
    return None

def construct():
    global _START_TIME, _DEADLINE
    if time.time() > _DEADLINE - 5:
        inner = _lattice_placement()
        return [[h[0], h[1], h[2]] for h in inner], (0.0, 0.0), 10.0, 0.0

    best_result = None
    best_side = float('inf')

    # Strategy 1: Try lattice-based initializations with Nelder-Mead
    lattice = _lattice_placement()
    if _validate_hexes(lattice):
        cx, cy, side, angle = _min_enclosing_side(lattice)
        if side < best_side:
            best_side = side
            best_result = ([[h[0], h[1], h[2]] for h in lattice], (cx, cy), side, angle)

    # Strategy 2: Nelder-Mead from lattice init
    if time.time() < _DEADLINE - 60:
        opt_hexes = _try_optimize_from_init(lattice, max_iter=400)
        if opt_hexes is not None:
            cx, cy, side, angle = _min_enclosing_side(opt_hexes)
            if side < best_side:
                best_side = side
                best_result = ([[h[0], h[1], h[2]] for h in opt_hexes], (cx, cy), side, angle)

    # Strategy 3: Greedy placement then optimize
    if time.time() < _DEADLINE - 50:
        greedy = _greedy_placement()
        if _validate_hexes(greedy):
            cx, cy, side, angle = _min_enclosing_side(greedy)
            if side < best_side:
                best_side = side
                best_result = ([[h[0], h[1], h[2]] for h in greedy], (cx, cy), side, angle)
            # Also try optimizing from greedy
            if time.time() < _DEADLINE - 40:
                opt_hexes = _try_optimize_from_init(greedy, max_iter=300)
                if opt_hexes is not None:
                    cx, cy, side, angle = _min_enclosing_side(opt_hexes)
                    if side < best_side:
                        best_side = side
                        best_result = ([[h[0], h[1], h[2]] for h in opt_hexes], (cx, cy), side, angle)

    # Strategy 4: Try different angle initializations
    if time.time() < _DEADLINE - 40:
        for base_angle in [0.0, 30.0, 60.0]:
            sqrt3 = math.sqrt(3)
            scale = 1.001
            u = (sqrt3 * scale, 0.0)
            v = (sqrt3 / 2 * scale, 1.5 * scale)
            cands = []
            for i in range(-3, 4):
                for j in range(-3, 4):
                    x = i * u[0] + j * v[0]
                    y = i * u[1] + j * v[1]
                    cands.append((x * x + y * y, x, y))
            cands.sort(key=lambda t: t[0])
            init_hexes = []
            for _, x, y in cands:
                init_hexes.append((x, y, base_angle))
                if len(init_hexes) == N:
                    break
            if _validate_hexes(init_hexes):
                cx, cy, side, angle = _min_enclosing_side(init_hexes)
                if side < best_side:
                    best_side = side
                    best_result = ([[h[0], h[1], h[2]] for h in init_hexes], (cx, cy), side, angle)
                if time.time() < _DEADLINE - 30:
                    opt_hexes = _try_optimize_from_init(init_hexes, max_iter=200)
                    if opt_hexes is not None:
                        cx, cy, side, angle = _min_enclosing_side(opt_hexes)
                        if side < best_side:
                            best_side = side
                            best_result = ([[h[0], h[1], h[2]] for h in opt_hexes], (cx, cy), side, angle)

    if best_result is not None:
        return best_result

    # Final fallback
    inner = _lattice_placement()
    return [[h[0], h[1], h[2]] for h in inner], (0.0, 0.0), 10.0, 0.0
