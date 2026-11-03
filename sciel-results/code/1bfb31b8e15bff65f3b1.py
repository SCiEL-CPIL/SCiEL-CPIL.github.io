import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
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
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True,
                                               'dual_feasibility_tolerance': 1e-12,
                                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None


def _smooth_outer_side(verts, outer_angle_deg, beta=60.0):
    """
    Smooth surrogate for the outer side length using log-sum-exp.
    For a given outer angle, the inradius is max over the 6 face widths/2.
    We approximate that max with a soft-max to get a differentiable objective.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    vdots = verts @ normals.T  # shape (M, 6)
    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)  # shape (6,)
    R_vals = widths / 2.0

    max_R = np.max(R_vals)
    shifted = R_vals - max_R
    log_sum_exp = max_R + np.log(np.sum(np.exp(beta * shifted))) / beta

    side = 2.0 * log_sum_exp * _INVSQRT3
    return side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    if layout_idx == 0:
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
        return positions[:N]
    elif layout_idx == 1:
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
        return positions[:N]
    elif layout_idx == 2:
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
        return positions[:N]
    elif layout_idx == 3:
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
        return positions[:N]
    elif layout_idx == 4:
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
        return positions[:N]
    elif layout_idx == 5:
        positions = []
        positions.append((0, 0))
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for ti, tj in ring1:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        ring2 = [(2, 0), (1, 1), (-1, 2), (-2, 0), (-1, -1), (0, -2)]
        for ti, tj in ring2[:4]:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        positions = []
        y1 = -0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-3, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        positions = []
        y1 = -0.75 * scale
        for k in range(-3, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        positions = []
        y0 = -3.0 * scale
        positions.append((0, y0))
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        y4 = 3.0 * scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return coords_sorted[:N]


def _support(a, theta):
    vals = np.cos(a + np.arange(6) * math.pi / 3.0 - theta)
    return np.max(vals)


def _hex_distance(p1, a1, p2, a2):
    """Minimum distance between two unit hexagons at positions p1, p2 with angles a1, a2 (radians)."""
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0

    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])

    h1 = _support(a1, theta)
    h2 = _support(a2, theta + math.pi)

    return d - h1 - h2


def _objective(params, positions, outer_angle_deg):
    """Objective: outer hexagon side length for given inner positions and angles."""
    n = len(positions)
    new_positions = []
    new_angles = []
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        a = params[3 * i + 2]
        new_positions.append((x, y))
        new_angles.append(a)

    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dist = _hex_distance(new_positions[i], new_angles[i],
                                 new_positions[j], new_angles[j])
            if dist < min_dist:
                min_dist = dist
            if dist < -1e-6:
                return 1000.0 + min_dist * 100

    if min_dist < 0:
        return 1000.0 + min_dist * 100

    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(new_positions[i][0], new_positions[i][1],
                              math.degrees(new_angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)

    cx, cy, side = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
    return side


def _refine_layout(positions, inner_angle, best_side, best_center, best_outer_angle,
                   outer_angles):
    """
    Refine a single layout: smooth surrogate over (positions, angles, outer_angle)
    then a high-precision exact LP polish. Returns (side, inner, center, outer_angle).
    """
    n = len(positions)
    if n == 0:
        return float('inf'), None, (0.0, 0.0), 0.0

    # Smooth surrogate objective over (x_i, y_i, a_i, outer_angle_deg)
    def smooth_obj(p):
        verts = np.empty((n * 6, 2))
        for i in range(n):
            verts[i * 6:(i + 1) * 6] = _hex_vertices(p[3 * i], p[3 * i + 1],
                                                     math.degrees(p[3 * i + 2]))
        return _smooth_outer_side(verts, p[-1])

    best_local_side = float('inf')
    best_local_inner = None
    best_local_center = (0.0, 0.0)
    best_local_angle = 0.0

    for oa in outer_angles:
        if time.time() > _DEADLINE - 15.0:
            break
        x0 = []
        for i in range(n):
            x0.append(positions[i][0])
            x0.append(positions[i][1])
            x0.append(math.radians(inner_angle))
        x0.append(oa)

        try:
            res = minimize(smooth_obj, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_local_side and res.fun < 100:
                new_positions = []
                new_angles = []
                for i in range(n):
                    new_positions.append((res.x[3 * i], res.x[3 * i + 1]))
                    new_angles.append(math.degrees(res.x[3 * i + 2]))
                oa_opt = res.x[-1]

                all_vertices = np.vstack(
                    [_hex_vertices(px, py, a) for px, py, a in
                     zip(new_positions, new_angles)])
                r = _min_enclosing_hex_side(all_vertices, oa_opt)
                if r is not None:
                    cx, cy, side = r
                    if side < best_local_side:
                        best_local_side = side
                        best_local_inner = [[px, py, a] for px, py, a in
                                            zip(new_positions, new_angles)]
                        best_local_center = (cx, cy)
                        best_local_angle = oa_opt
        except Exception:
            pass

    if best_local_side < best_side:
        return best_local_side, best_local_inner, best_local_center, best_local_angle
    return best_side, best_inner_global, best_center, best_outer_angle


best_inner_global = None


def construct():
    global _START_TIME, _DEADLINE, best_inner_global

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]

    inner_angle = 30.0
    # Coarse sweep of outer angles for the initial (unoptimized) layout
    coarse_angles = list(np.arange(0, 30.0, 0.25))

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 25.0:
            break

        all_vertices = np.vstack(
            [_hex_vertices(px, py, inner_angle) for px, py in positions])

        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0

        for oa in coarse_angles:
            if time.time() > _DEADLINE - 25.0:
                break
            r = _min_enclosing_hex_side(all_vertices, oa)
            if r is not None:
                cx, cy, side = r
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = oa

        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle

    # Refinement: smooth surrogate + exact polish, per layout, using the
    # coarse best outer angle plus a few neighbours as starting points.
    best_inner_global = best_inner
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        for layout_idx, positions in enumerate(layouts):
            if time.time() > _DEADLINE - 20.0:
                break
            oa_starts = [best_outer_angle, best_outer_angle + 5,
                         best_outer_angle - 5, best_outer_angle + 15]
            side, inner, center, oa = _refine_layout(
                positions, inner_angle, best_side, best_center,
                best_outer_angle, oa_starts)
            if inner is not None and side < best_side:
                best_side = side
                best_inner = inner
                best_center = center
                best_outer_angle = oa

    # Final high-precision polish: tight-tolerance LP over a fine angle grid
    # around the current best outer angle, with a couple of restarts.
    if best_inner is not None and time.time() < _DEADLINE - 10.0:
        n = len(best_inner)
        all_vertices = np.vstack(
            [_hex_vertices(px, py, a) for px, py, a in best_inner])
        fine_angles = np.arange(best_outer_angle - 2.0, best_outer_angle + 2.0, 0.05)
        for oa in fine_angles:
            if time.time() > _DEADLINE - 10.0:
                break
            r = _min_enclosing_hex_side(all_vertices, oa)
            if r is not None:
                cx, cy, side = r
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_outer_angle = oa

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
