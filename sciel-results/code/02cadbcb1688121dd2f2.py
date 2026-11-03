import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)

    A_rows = []
    b_vals = []

    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])

    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-14,
                               'primal_feasibility_tolerance': 1e-14})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None


def _is_valid_packing(positions, angles, side=1.0, eps=1e-9):
    """Check that all hexagons are disjoint (distance between centers >= 2*side)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if dx * dx + dy * dy < (2.0 * side - eps) ** 2:
                return False
    return True


def _smooth_side(verts, outer_angle_deg, beta=200.0):
    """
    Smooth surrogate for the outer side length.
    Returns scalar value (no gradient).
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    centered = verts - np.mean(verts, axis=0)
    vdots = centered @ normals.T  # shape (M, 6)

    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)  # shape (6,)
    R_vals = widths / 2.0

    max_R = np.max(R_vals)
    shifted = R_vals - max_R
    log_sum_exp = max_R + np.log(np.sum(np.exp(beta * shifted))) / beta

    side = 2.0 * log_sum_exp * _INVSQRT3
    return side


def _build_initial_layouts():
    """Generate multiple initial layouts for 11 hexagons."""
    layouts = []
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    # Layout 0: compact 1-6-4
    target = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    target.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    target.extend(ring2_sorted[:4])
    positions = []
    for ti, tj in target:
        x = ti * u[0] + tj * v[0]
        y = ti * u[1] + tj * v[1]
        positions.append((x, y))
    layouts.append(positions[:N])

    # Layout 1: 3-4-4 rows
    positions = []
    y1 = -1.5 * scale
    for k in range(-1, 2):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 3):
        x = k * sqrt3 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-1, 3):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y3))
    layouts.append(positions[:N])

    # Layout 2: 4-3-4 rows
    positions = []
    y1 = -1.5 * scale
    for k in range(-2, 2):
        x = k * sqrt3 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 2):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-2, 2):
        x = k * sqrt3 * scale
        positions.append((x, y3))
    layouts.append(positions[:N])

    # Layout 3: 3-5-3 rows
    positions = []
    y1 = -1.5 * scale
    for k in range(-1, 2):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-2, 3):
        x = k * sqrt3 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-1, 2):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y3))
    layouts.append(positions[:N])

    # Layout 4: 2-4-2-3 staggered
    positions = []
    y1 = -3.0 * scale
    for k in range(0, 2):
        x = k * sqrt3 * scale - sqrt3 / 2 * scale
        positions.append((x, y1))
    y2 = -1.5 * scale
    for k in range(-1, 3):
        x = k * sqrt3 * scale
        positions.append((x, y2))
    y3 = 0.0
    for k in range(-1, 1):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y3))
    y4 = 1.5 * scale
    for k in range(-1, 2):
        x = k * sqrt3 * scale
        positions.append((x, y4))
    layouts.append(positions[:N])

    # Layout 5: 2-3-4-2
    positions = []
    y1 = -3.0 * scale
    for k in range(0, 2):
        x = k * sqrt3 * scale - sqrt3 / 2 * scale
        positions.append((x, y1))
    y2 = -1.5 * scale
    for k in range(-1, 2):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y2))
    y3 = 0.0
    for k in range(-2, 2):
        x = k * sqrt3 * scale
        positions.append((x, y3))
    y4 = 1.5 * scale
    for k in range(0, 2):
        x = k * sqrt3 * scale + sqrt3 / 2 * scale
        positions.append((x, y4))
    layouts.append(positions[:N])

    return layouts


def _optimize_joint(positions_init, angles_init, time_limit):
    """
    Jointly optimize inner positions and outer hexagon (center, side, angle).
    Uses smooth surrogate + Nelder-Mead + final LP polish.
    Returns (best_side, best_positions, best_angles, best_center, best_outer_angle).
    """
    n = len(positions_init)

    def decode(x):
        # x = [x1,y1,...,xn,yn, theta1,...,thetan, outer_angle]
        pos = np.array([(x[i] - x[i + n], x[i + n] - x[i + 2 * n]) for i in range(n)])
        ang = x[3 * n:3 * n + n]
        outer_a = x[3 * n + n]
        return pos, ang, outer_a

    def objective(x):
        pos, ang, outer_a = decode(x)
        # Displacement penalty for too close hexagons
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d2 = dx * dx + dy * dy
                if d2 < 4.0:
                    penalty += (2.0 - math.sqrt(d2)) ** 2
        # Build all vertices
        verts = np.vstack([_hex_vertices(pos[i, 0], pos[i, 1], ang[i]) for i in range(n)])
        side = _smooth_side(verts, outer_a, beta=200.0)
        return side + 50.0 * penalty

    # Initial x
    x0 = []
    for i in range(n):
        x0.extend([positions_init[i][0], positions_init[i][1]])
    for i in range(n):
        x0.extend([angles_init[i]])
    x0.append(0.0)

    best_side = 1e10
    best_pos = np.array(positions_init, dtype=float)
    best_ang = np.array(angles_init, dtype=float)
    best_outer_a = 0.0
    best_center = (0.0, 0.0)

    # Nelder-Mead on smooth objective
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True, 'initial_simplex': 0.1})
        if time.time() < time_limit:
            pos, ang, outer_a = decode(res.x)
            # Check validity
            if _is_valid_packing([(pos[i, 0], pos[i, 1]) for i in range(n)],
                                 [ang[i] for i in range(n)]):
                verts = np.vstack([_hex_vertices(pos[i, 0], pos[i, 1], ang[i]) for i in range(n)])
                center_off = np.mean(verts, axis=0)
                centered = verts - center_off
                # Grid search for outer angle
                coarse_best = 1e10
                coarse_best_a = 0.0
                for oa in np.arange(0, 30.25, 0.25):
                    if time.time() > time_limit:
                        break
                    r = _min_enclosing_hex_side(centered, oa)
                    if r:
                        if r[2] < coarse_best:
                            coarse_best = r[2]
                            coarse_best_a = oa
                if coarse_best < best_side:
                    best_side = coarse_best
                    best_pos = pos.copy()
                    best_ang = ang.copy()
                    best_outer_a = coarse_best_a
                    # Final LP for center
                    r = _min_enclosing_hex_side(centered, coarse_best_a)
                    if r:
                        best_center = (r[0] + center_off[0], r[1] + center_off[1])
    except Exception:
        pass

    # If we have a valid result, do a final high-precision polish
    if best_side < 1e9 and time.time() < time_limit:
        verts = np.vstack([_hex_vertices(best_pos[i, 0], best_pos[i, 1], best_ang[i]) for i in range(n)])
        center_off = np.mean(verts, axis=0)
        centered = verts - center_off
        # Fine grid around best_outer_a
        fine_best = best_side
        fine_best_a = best_outer_a
        for oa in np.arange(max(0, best_outer_a - 1.0), min(30, best_outer_a + 1.0) + 0.01, 0.01):
            if time.time() > time_limit:
                break
            r = _min_enclosing_hex_side(centered, oa)
            if r:
                if r[2] < fine_best:
                    fine_best = r[2]
                    fine_best_a = oa
        if fine_best < best_side:
            best_side = fine_best
            best_outer_a = fine_best_a
            r = _min_enclosing_hex_side(centered, fine_best_a)
            if r:
                best_center = (r[0] + center_off[0], r[1] + center_off[1])

    return best_side, best_pos, best_ang, best_center, best_outer_a


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    layouts = _build_initial_layouts()
    angle_sets = [
        [0.0] * N,
        [30.0] * N,
        [0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0],
        [15.0] * N,
    ]

    time_per_attempt = 12.0
    attempts = 0
    max_attempts = 8

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        for angles in angle_sets:
            if time.time() > _DEADLINE - 15.0:
                break
            if attempts >= max_attempts:
                break
            attempts += 1
            time_limit = min(_DEADLINE - 15.0, time.time() + time_per_attempt)
            try:
                side, pos, ang, center, outer_a = _optimize_joint(
                    positions, angles, time_limit)
                if side < best_side:
                    best_side = side
                    best_inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])]
                                  for i in range(N)]
                    best_center = center
                    best_outer_angle = outer_a
            except Exception:
                pass

    # Final polish: try to improve outer angle with tighter search on best result
    if best_inner is not None and time.time() < _DEADLINE - 5.0:
        verts = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
        center_off = np.mean(verts, axis=0)
        centered = verts - center_off
        # Fine grid around best_outer_angle
        fine_best = best_side
        fine_best_a = best_outer_angle
        for oa in np.arange(max(0, best_outer_angle - 2.0),
                            min(30, best_outer_angle + 2.0) + 0.01, 0.01):
            if time.time() > _DEADLINE - 3.0:
                break
            r = _min_enclosing_hex_side(centered, oa)
            if r:
                if r[2] < fine_best:
                    fine_best = r[2]
                    fine_best_a = oa
        if fine_best < best_side:
            best_side = fine_best
            best_outer_angle = fine_best_a
            r = _min_enclosing_hex_side(centered, fine_best_a)
            if r:
                best_center = (r[0] + center_off[0], r[1] + center_off[1])

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
