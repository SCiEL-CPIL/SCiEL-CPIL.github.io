"""Combined program: LP outer-hexagon minimization + local refinement with disjointness checks."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin

def _lp_options():
    return {'presolve': True, 'time_limit': 8.0,
            'dual_feasibility_tolerance': 1e-10,
            'primal_feasibility_tolerance': 1e-10}

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _on_segment(a, b, p, eps=1e-12):
    """Check if point p is on segment a-b."""
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)

    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True

    if abs(c1) < 1e-12 and _on_segment(p3, p4, p1):
        return True
    if abs(c2) < 1e-12 and _on_segment(p3, p4, p2):
        return True
    if abs(c3) < 1e-12 and _on_segment(p1, p2, p3):
        return True
    if abs(c4) < 1e-12 and _on_segment(p1, p2, p4):
        return True
    return False

def _point_in_convex_polygon(p, polygon, strict=False):
    """Check if point p is inside convex polygon."""
    px, py = p
    n = len(polygon)
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if strict:
            if cross < 0:
                return False
        else:
            if cross < -1e-12:
                return False
        if cross > 0:
            if sign < 0:
                return False
            sign = 1
        elif cross < 0:
            if sign > 0:
                return False
            sign = -1
    return True

def _hexagons_overlap(verts1, verts2):
    """Check if two convex hexagons overlap (interior intersection)."""
    if _point_in_convex_polygon(verts1[0], verts2, strict=True):
        return True
    if _point_in_convex_polygon(verts2[0], verts1, strict=True):
        return True

    for k in range(6):
        a1 = verts1[k]
        b1 = verts1[(k + 1) % 6]
        for l in range(6):
            a2 = verts2[l]
            b2 = verts2[(l + 1) % 6]
            if _segments_intersect(a1, b1, a2, b2):
                return True
    return False

def _check_disjoint(inner_hexes, tol=1e-9):
    """Check that all inner hexagons are disjoint."""
    n = len(inner_hexes)
    all_verts = []
    for ix, iy, iangle in inner_hexes:
        all_verts.append(_get_hex_vertices(ix, iy, iangle))

    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(all_verts[i], all_verts[j]):
                return False
    return True

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """
    For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side, angle_deg).
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)

    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg

    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(vx * nx + vy * ny)

    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(None, None), (None, None), (0, None)],
                         method='highs', options=_lp_options())
        if result.success:
            cx_opt = result.x[0]
            cy_opt = result.x[1]
            r_opt = result.x[2]
            side = r_opt / (math.sqrt(3) / 2)
            return cx_opt, cy_opt, side, angle_deg
    except Exception:
        pass

    # Fallback: Centroid based
    cx_mean = sum(v[0] for v in all_vertices) / n_pts
    cy_mean = sum(v[1] for v in all_vertices) / n_pts
    r_opt = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
            r_opt = max(r_opt, proj)
            neg_proj = -(vx - cx_mean) * nx - (vy - cy_mean) * ny
            r_opt = max(r_opt, neg_proj)
    side = r_opt / (math.sqrt(3) / 2)
    return cx_mean, cy_mean, side, angle_deg

def _objective(angle_params, inner_hexes):
    """Objective function for Nelder-Mead: minimize side length for a given angle."""
    angle_deg = angle_params[0]
    if time.time() > _DEADLINE - 2.0:
        return 1e9
    try:
        _, _, side, _ = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        return side
    except Exception:
        return 1e9

def _find_best_outer_lp(inner_hexes):
    """
    Find the best outer hexagon by optimizing the angle using Nelder-Mead.
    """
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)

    angles_coarse = np.arange(0, 30.1, 0.25)
    for angle_deg in angles_coarse:
        if time.time() > _DEADLINE - 5.0:
            break
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)

    x0 = np.array([best_angle])

    def obj_closure(x):
        return _objective(x, inner_hexes)

    try:
        res = minimize(obj_closure, x0, method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True})
        if res.fun < best_side:
            best_angle = res.x[0]
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
            best_side = side
            best_center = (cx, cy)
        else:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
            best_side = side
            best_center = (cx, cy)
    except Exception:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
        best_side = side
        best_center = (cx, cy)

    return best_center[0], best_center[1], best_side, best_angle

def _refine_positions(inner_hexes, max_iter=100):
    """
    Lightweight local refinement of inner positions and angles using Nelder-Mead,
    with disjointness constraint and outer-hexagon side as objective.
    Returns improved inner_hexes or None if no improvement found.
    """
    n = len(inner_hexes)
    # Flatten: x0, y0, a0, x1, y1, a1, ...
    x0 = []
    for ix, iy, ia in inner_hexes:
        x0.extend([ix, iy, ia])
    x0 = np.array(x0, dtype=float)

    def objective(params):
        if time.time() > _DEADLINE - 3.0:
            return 1e9
        # Unpack
        pos = [(params[3*i], params[3*i+1]) for i in range(n)]
        ang = [params[3*i+2] for i in range(n)]
        inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(n)]

        # Check disjointness
        if not _check_disjoint(inner):
            return 1e6

        # Evaluate outer side
        try:
            best_side = float('inf')
            # Coarse sweep over outer angles
            for outer_angle in [0.0, 15.0, 30.0]:
                _, _, side, _ = _min_enclosing_hex_side_lp(inner, outer_angle)
                if side < best_side:
                    best_side = side
            return best_side
        except Exception:
            return 1e9

    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
        if res.fun < 1e5:  # Valid solution found
            new_inner = []
            for i in range(n):
                new_inner.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
            # Final validation
            if _check_disjoint(new_inner):
                return new_inner
    except Exception:
        pass

    return None

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    sqrt3 = math.sqrt(3)

    def _generate_positions(scale):
        """Generate hexagonal lattice positions near origin."""
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)

        candidates = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))

        candidates.sort(key=lambda t: t[0])

        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        return positions

    # Try multiple scales to find better packings
    scales = [1.0005, 1.001, 1.002, 1.005]
    all_position_sets = []
    for s in scales:
        positions = _generate_positions(s)
        if positions:
            all_position_sets.append(positions)

    # Use the first set as default
    positions = all_position_sets[0] if all_position_sets else []

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try multiple position sets and inner angles
    position_sets_to_try = all_position_sets[:3]

    for pos_set in position_sets_to_try:
        if time.time() > _DEADLINE - 25.0:
            break

        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 20.0:
                break

            inner = [[pos[0], pos[1], inner_angle] for pos in pos_set]

            if time.time() > _DEADLINE - 12.0:
                # Fast fallback for this configuration if running out of time
                cx, cy, side, ang = _min_enclosing_hex_side_lp(inner, 0.0)
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_center = (cx, cy)
                    best_angle = ang
                continue

            cx, cy, side, angle = _find_best_outer_lp(inner)

            if side < best_side:
                best_side = side
                best_inner = inner
                best_center = (cx, cy)
                best_angle = angle

    # New: Try local refinement on the best configuration found
    if best_inner is not None and time.time() < _DEADLINE - 15.0:
        refined_inner = _refine_positions(best_inner, max_iter=80)
        if refined_inner is not None:
            # Evaluate the refined configuration
            cx, cy, side, angle = _find_best_outer_lp(refined_inner)
            if side < best_side - 1e-10:
                best_side = side
                best_inner = refined_inner
                best_center = (cx, cy)
                best_angle = angle

    # Final polish: refine the outer hexagon angle with tighter tolerances
    if best_inner is not None and time.time() < _DEADLINE - 8.0:
        try:
            x0 = np.array([best_angle])
            def polish_obj(x):
                if time.time() > _DEADLINE - 3.0:
                    return 1e9
                _, _, side, _ = _min_enclosing_hex_side_lp(best_inner, float(x[0]))
                return side

            # First pass: moderate tolerance
            res_polish1 = minimize(polish_obj, x0, method='Nelder-Mead',
                                 options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 300, 'adaptive': True})

            best_angle_p1 = res_polish1.x[0]
            best_side_p1 = res_polish1.fun

            # Second pass: very tight tolerance from refined starting point
            if time.time() < _DEADLINE - 5.0:
                x0_p2 = np.array([best_angle_p1])
                res_polish2 = minimize(polish_obj, x0_p2, method='Nelder-Mead',
                                     options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 200, 'adaptive': True})

                if res_polish2.fun < best_side_p1 - 1e-12:
                    best_angle_p1 = res_polish2.x[0]
                    best_side_p1 = res_polish2.fun

            if best_side_p1 < best_side - 1e-12:
                best_angle = best_angle_p1
                cx, cy, side, ang = _min_enclosing_hex_side_lp(best_inner, best_angle)
                best_side = side
                best_center = (cx, cy)
        except Exception:
            pass

    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle
