import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Minimum side length of a regular hexagon (vertex at outer_angle_deg)
    containing all points in verts. Vectorized LP over center + inradius.
    Returns (cx, cy, side).
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    v_dot_n = verts @ normals.T  # (M, 6)

    A_rows = []
    b_vals = []
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        # (v - c).n <= R  ->  -n.c - R <= -v.n
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        # (c - v).n <= R  ->   n.c - R <=  v.n
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass

    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0:
        r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

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
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return coords_sorted[:N]

def _hex_distance(p1, a1, p2, a2):
    """Minimum distance between two unit hexagons at (p1,a1),(p2,a2) (a in radians)."""
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])

    def support(a, theta):
        vals = np.cos(a + np.arange(6) * math.pi / 3.0 - theta)
        return np.max(vals)

    h1 = support(a1, theta)
    h2 = support(a2, theta + math.pi)
    return d - h1 - h2

def _pairwise_softmin(params, n, beta):
    """Soft-min (log-sum-exp) of all pairwise hex distances; smooth in params."""
    dmin = float('inf')
    for i in range(n):
        pi = (params[3 * i], params[3 * i + 1])
        ai = params[3 * i + 2]
        for j in range(i + 1, n):
            pj = (params[3 * j], params[3 * j + 1])
            aj = params[3 * j + 2]
            dd = _hex_distance(pi, ai, pj, aj)
            if dd < dmin:
                dmin = dd
    # soft-min of the 55 distances approximates min; use dmin as anchor
    # recompute vector of distances for the log-sum-exp
    vals = []
    for i in range(n):
        pi = (params[3 * i], params[3 * i + 1])
        ai = params[3 * i + 2]
        for j in range(i + 1, n):
            pj = (params[3 * j], params[3 * j + 1])
            aj = params[3 * j + 2]
            vals.append(_hex_distance(pi, ai, pj, aj))
    vals = np.array(vals)
    m = np.min(vals)
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def _objective(params, outer_angle_deg):
    """Objective: outer hexagon side length; heavy penalty if any overlap."""
    n = N
    if time.time() > _DEADLINE - 2.0:
        return 1e6
    # overlap check via soft-min (cheap-ish, smooth)
    sm = _pairwise_softmin(params, n, 40.0)
    if sm < 0:
        return 1000.0 - sm * 100
    all_vertices = np.vstack([
        _hex_vertices(params[3 * i], params[3 * i + 1], math.degrees(params[3 * i + 2]))
        for i in range(n)
    ])
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side

def _objective_smooth(params, outer_angle_deg, beta=80.0):
    """Smooth surrogate: side length + soft-overlap penalty (epigraph-style)."""
    n = N
    if time.time() > _DEADLINE - 2.0:
        return 1e6
    sm = _pairwise_softmin(params, n, beta)
    all_vertices = np.vstack([
        _hex_vertices(params[3 * i], params[3 * i + 1], math.degrees(params[3 * i + 2]))
        for i in range(n)
    ])
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side + max(0.0, -sm) * 10.0

def _two_stage_angle(all_vertices, base_angle, coarse_step=1.0, fine_step=0.05):
    """Coarse scan then fine window around best outer angle."""
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    for outer_angle_deg in np.arange(0, 30.1, coarse_step):
        if time.time() > _DEADLINE - 8.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = outer_angle_deg
                best_center = (cx, cy)
    for outer_angle_deg in np.arange(max(0.0, best_angle - coarse_step),
                                     min(30.0, best_angle + coarse_step), fine_step):
        if time.time() > _DEADLINE - 8.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = outer_angle_deg
                best_center = (cx, cy)
    return best_side, best_angle, best_center

def construct():
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None

    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 12.0:
            break

        # try both inner orientations (flat-top 0 and pointy 30)
        for inner_angle in (0.0, 30.0):
            if time.time() > _DEADLINE - 12.0:
                break
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            side, angle, center = _two_stage_angle(all_vertices, 0.0)
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = center
                best_outer_angle = angle
                best_params = []
                for pos in positions:
                    best_params.append(pos[0])
                    best_params.append(pos[1])
                    best_params.append(math.radians(inner_angle))

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Phase 1: L-BFGS-B local optimization on the best layout
    if time.time() < _DEADLINE - 30.0 and best_params is not None:
        x0 = best_params[:]
        outer_angles_to_try = list(np.arange(max(0.0, best_outer_angle - 10),
                                             min(30.0, best_outer_angle + 10), 1.0))
        for outer_angle_deg in outer_angles_to_try:
            if time.time() > _DEADLINE - 30.0:
                break
            try:
                res = minimize(_objective, x0, args=(outer_angle_deg,),
                               method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_side and res.fun < 100:
                    new_positions = [(res.x[3 * i], res.x[3 * i + 1]) for i in range(N)]
                    new_angles = [math.degrees(res.x[3 * i + 2]) for i in range(N)]
                    all_vertices = np.vstack([_hex_vertices(px, py, a)
                                              for px, py, a in zip(new_positions, new_angles)])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_outer_angle = outer_angle_deg
                        best_params = res.x[:]
            except Exception:
                pass

    # Phase 2: smooth surrogate local search (soft-min epigraph)
    if time.time() < _DEADLINE - 15.0 and best_params is not None:
        x0 = best_params[:]
        for beta in (100.0, 50.0):
            if time.time() > _DEADLINE - 15.0:
                break
            for outer_angle_deg in (best_outer_angle, best_outer_angle + 1.0, best_outer_angle - 1.0):
                if time.time() > _DEADLINE - 15.0:
                    break
                try:
                    res = minimize(_objective_smooth, x0, args=(outer_angle_deg, beta),
                                   method='L-BFGS-B',
                                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                    if res.fun < best_side and res.fun < 100:
                        new_positions = [(res.x[3 * i], res.x[3 * i + 1]) for i in range(N)]
                        new_angles = [math.degrees(res.x[3 * i + 2]) for i in range(N)]
                        all_vertices = np.vstack([_hex_vertices(px, py, a)
                                                  for px, py, a in zip(new_positions, new_angles)])
                        cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                        if side < best_side:
                            best_side = side
                            best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                            best_center = (cx, cy)
                            best_params = res.x[:]
                except Exception:
                    pass

    # Phase 3: final high-precision polish
    if time.time() < _DEADLINE - 5.0 and best_params is not None:
        x0 = best_params[:]
        for _ in range(5):
            if time.time() > _DEADLINE - 5.0:
                break
            try:
                res = minimize(_objective, x0, args=(best_outer_angle,),
                               method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
                if res.fun < best_side and res.fun < 100:
                    new_positions = [(res.x[3 * i], res.x[3 * i + 1]) for i in range(N)]
                    new_angles = [math.degrees(res.x[3 * i + 2]) for i in range(N)]
                    all_vertices = np.vstack([_hex_vertices(px, py, a)
                                              for px, py, a in zip(new_positions, new_angles)])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_params = res.x[:]
                    else:
                        break
                else:
                    break
            except Exception:
                break

    return best_inner, best_center, best_side, best_outer_angle
