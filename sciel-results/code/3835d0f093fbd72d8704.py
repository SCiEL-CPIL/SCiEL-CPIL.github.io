"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses smooth optimization with log-sum-exp surrogate for the min-enclosing hexagon,
plus a smooth non-overlap penalty, with multi-start and L-BFGS-B.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE_MARGIN = 15.0  # seconds before the 150s limit
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon (with small tolerance)."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-7:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-8):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _all_disjoint(inner_hexes):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _support_function(cx, cy, s, ang_deg, nx, ny):
    """Support function of a hexagon in direction (nx, ny)."""
    max_val = 0.0
    for k in range(6):
        a = math.radians(ang_deg + 60 * k)
        vx = s * math.cos(a)
        vy = s * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _min_enclosing_hex_side_lp(inner_hexes, oangle=0.0):
    """Find minimum side length of outer hexagon using LP. Returns (center, side)."""
    c = [0, 0, 1]  # minimize t (apothem)
    A_ub = []
    b_ub = []

    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)

    A_ub.append([0, 0, -1])
    b_ub.append(0)

    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')

    if res.success:
        t = res.x[2]
        side = t * 2.0 / SQRT3
        center = (res.x[0], res.x[1])
        return center, side
    else:
        # Fallback: centroid
        cx_mean = sum(h[0] for h in inner_hexes) / len(inner_hexes)
        cy_mean = sum(h[1] for h in inner_hexes) / len(inner_hexes)
        center = (cx_mean, cy_mean)
        # Compute max distance to determine side
        max_t = 0
        for cx_h, cy_h, s_h, ang_h in inner_hexes:
            verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
            for vx, vy in verts:
                for k in range(6):
                    nx = math.cos(math.radians(oangle + 30 + 60 * k))
                    ny = math.sin(math.radians(oangle + 30 + 60 * k))
                    val = (vx - cx_mean) * nx + (vy - cy_mean) * ny
                    if val > max_t:
                        max_t = val
        side = max_t * 2.0 / SQRT3
        return center, side


def _smooth_objective(state, oangle, n_hex):
    """
    Smooth objective: outer hex side via log-sum-exp + non-overlap penalty.
    state = [x1, y1, a1, x2, y2, a2, ..., xn, yn, an]
    """
    # Build inner hexes
    hexes = []
    for i in range(n_hex):
        hexes.append((state[3*i], state[3*i+1], 1.0, state[3*i+2]))

    # Compute outer hex via LP (exact)
    center, side = _min_enclosing_hex_side_lp(hexes, oangle)

    # Non-overlap penalty using support functions
    penalty = 0.0
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            max_violation = 0.0
            for k in range(12):
                theta = math.radians(15 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)

                s_i = _support_function(state[3*i], state[3*i+1], 1.0, state[3*i+2], nx, ny)
                s_j = _support_function(state[3*j], state[3*j+1], 1.0, state[3*j+2], -nx, -ny)

                dx = state[3*j] - state[3*i]
                dy = state[3*j+1] - state[3*i+1]
                dist_proj = dx * nx + dy * ny

                violation = s_i + s_j - dist_proj
                if violation > max_violation:
                    max_violation = violation

            if max_violation > 0:
                penalty += max_violation ** 2

    return side + 1000.0 * penalty


def _axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian."""
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)


def _lattice_initial(n=11):
    """Generate initial lattice positions for n hexagons."""
    axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    axial.extend(ring1)
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    axial.extend(ring2[:n - 7])
    xy = [_axial_to_xy(q, r) for q, r in axial[:n]]
    return xy


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Try lattice configurations with various outer angles
    init_xy = _lattice_initial(N)
    for oangle in [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0, 10.0, -10.0]:
        if time.time() > deadline:
            break
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in init_xy]
        center, side = _min_enclosing_hex_side_lp(inner_hexes, oangle)
        if side < best_side:
            best_side = side
            best_positions = init_xy[:]
            best_angles = [0.0] * N
            best_center = center
            best_oangle = oangle

    if best_positions is None:
        best_positions = init_xy
        best_angles = [0.0] * N
        best_center, best_side = _min_enclosing_hex_side_lp(
            [(x, y, 1.0, 0.0) for x, y in init_xy], 0.0)
        best_oangle = 0.0

    # Phase 2: Multi-start L-BFGS-B optimization
    n_hex = N
    n_vars = 3 * n_hex  # x, y, angle for each hex

    # Precompute direction vectors for overlap checks
    dir_angles = [math.radians(15 * k) for k in range(12)]
    dir_coss = [math.cos(a) for a in dir_angles]
    dir_sins = [math.sin(a) for a in dir_angles]

    # Precompute hexagon vertex offsets for each angle (we'll recompute as needed)

    def objective(state):
        hexes = []
        for i in range(n_hex):
            hexes.append((state[3*i], state[3*i+1], 1.0, state[3*i+2]))

        # Compute outer hex side via LP
        center, side = _min_enclosing_hex_side_lp(hexes, best_oangle)

        # Non-overlap penalty
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                max_violation = 0.0
                for k in range(12):
                    nx = dir_coss[k]
                    ny = dir_sins[k]

                    s_i = _support_function(state[3*i], state[3*i+1], 1.0, state[3*i+2], nx, ny)
                    s_j = _support_function(state[3*j], state[3*j+1], 1.0, state[3*j+2], -nx, -ny)

                    dx = state[3*j] - state[3*i]
                    dy = state[3*j+1] - state[3*i+1]
                    dist_proj = dx * nx + dy * ny

                    violation = s_i + s_j - dist_proj
                    if violation > max_violation:
                        max_violation = violation

                if max_violation > 0:
                    penalty += max_violation ** 2

        return side + 1000.0 * penalty

    # Multi-start
    starts = []
    # Start 1: lattice
    starts.append(np.array([v for i in range(n_hex) for v in
                            [best_positions[i][0], best_positions[i][1], 0.0]]))
    # Start 2: slightly perturbed lattice
    np.random.seed(42)
    for _ in range(5):
        perturbed = []
        for i in range(n_hex):
            px = best_positions[i][0] + np.random.uniform(-0.3, 0.3)
            py = best_positions[i][1] + np.random.uniform(-0.3, 0.3)
            pa = np.random.uniform(-30, 30)
            perturbed.extend([px, py, pa])
        starts.append(np.array(perturbed))
    # Start 3: more spread out
    for _ in range(3):
        spread = []
        for i in range(n_hex):
            angle = 2 * math.pi * i / n_hex
            r = 2.5 + np.random.uniform(-0.5, 0.5)
            px = r * math.cos(angle) + np.random.uniform(-0.2, 0.2)
            py = r * math.sin(angle) + np.random.uniform(-0.2, 0.2)
            pa = np.random.uniform(-60, 60)
            spread.extend([px, py, pa])
        starts.append(np.array(spread))

    for s_idx, init_state in enumerate(starts):
        if time.time() > deadline:
            break

        try:
            result = minimize(objective, init_state, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            continue

        final_state = result.x
        hexes = []
        for i in range(n_hex):
            hexes.append((final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]))

        # Verify disjointness
        if _all_disjoint(hexes):
            center, side = _min_enclosing_hex_side_lp(hexes, best_oangle)
            if side < best_side:
                best_side = side
                best_positions = [(h[0], h[1]) for h in hexes]
                best_angles = [h[3] for h in hexes]
                best_center = center

    # Phase 3: Polish with tighter optimization on best configuration
    if time.time() < deadline - 10:
        init_state = np.array([v for i in range(n_hex) for v in
                               [best_positions[i][0], best_positions[i][1], best_angles[i]]])
        try:
            result = minimize(objective, init_state, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        except Exception:
            result = None

        if result is not None:
            final_state = result.x
            hexes = []
            for i in range(n_hex):
                hexes.append((final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]))
            if _all_disjoint(hexes):
                center, side = _min_enclosing_hex_side_lp(hexes, best_oangle)
                if side < best_side:
                    best_side = side
                    best_positions = [(h[0], h[1]) for h in hexes]
                    best_angles = [h[3] for h in hexes]
                    best_center = center

    # Phase 4: Try different outer angles with best inner configuration
    if time.time() < deadline - 5:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                       -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline:
                break
            inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                          for i in range(n_hex)]
            center, side = _min_enclosing_hex_side_lp(inner_hexes, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Final verification
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                  for i in range(n_hex)]

    if not _all_disjoint(inner_hexes):
        # Scale up positions slightly
        for scale in [1.001, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _min_enclosing_hex_side_lp(inner_hexes, best_oangle)
                break

    # Verify containment
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side until valid
        for _ in range(200):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    # Format output
    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
            for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
