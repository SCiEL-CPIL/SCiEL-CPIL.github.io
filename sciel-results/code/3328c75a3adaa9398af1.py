"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 148

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Discrete grid for barycentric coordinates
GRID_SIZE = 40  # 40x40 grid in barycentric (b, c) space


def _cartesian_to_barycentric(pts):
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])


def _barycentric_to_cartesian(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])


def _project_to_triangle(pts):
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.min(np.abs(cross)) * 0.5


def _all_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta):
    areas = _all_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    return max_a - np.log(np.sum(np.exp(shifted))) / beta


def _soft_min_grad(pts, beta):
    areas = _all_areas(pts)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0
    w = 0.5 * weights * sign_cross
    grad = np.zeros((N, 2))
    grad_pi = w[:, None] * np.column_stack([-dy2, dx2])
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])
    grad_pk = w[:, None] * np.column_stack([-dy1, dx1])
    np.add.at(grad, _T_I, grad_pi)
    np.add.at(grad, _T_J, grad_pj)
    np.add.at(grad, _T_K, grad_pk)
    return grad


def _bary_obj(flat_bc, beta):
    bc = flat_bc.reshape(-1, 2)
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1 - b)
    pts = _barycentric_to_cartesian(np.column_stack([b, c]))
    val = -_soft_min_area(pts, beta)
    if beta > 0:
        grad_pts = _soft_min_grad(pts, beta)
        grad_b = grad_pts[:, 0]
        grad_c = 0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1]
        grad_flat = np.column_stack([-grad_b, -grad_c]).flatten()
    else:
        grad_flat = np.zeros_like(flat_bc)
    return val, grad_flat


def _grid_search_discrete(n_starts=50, grid_size=40):
    """Search on a discrete grid of barycentric coordinates."""
    rng = np.random.default_rng(42)
    best_pts = None
    best_area = 0.0
    
    # Generate grid points in barycentric space
    b_vals = np.linspace(0, 1, grid_size)
    c_vals = np.linspace(0, 1, grid_size)
    
    for start in range(n_starts):
        # Random initial configuration on grid
        bc = np.zeros((N, 2))
        for i in range(N):
            b_idx = rng.integers(0, grid_size)
            c_idx = rng.integers(0, grid_size)
            b = b_vals[b_idx]
            c = c_vals[c_idx]
            if b + c > 1:
                # Reflect to valid region
                b, c = 1 - b, 1 - c
            bc[i] = [b, c]
        
        pts = _barycentric_to_cartesian(bc)
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    return best_pts, best_area


def _local_refinement_grid(pts, n_iter=100, grid_size=20):
    """Local refinement around a given point using discrete grid moves."""
    bc = _cartesian_to_barycentric(pts)
    best_bc = bc.copy()
    best_area = _min_triangle_area(pts)
    
    # Grid step sizes
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for step in step_sizes:
        for _ in range(n_iter):
            improved = False
            # Try moving each point in 8 directions on a local grid
            for i in range(N):
                b0, c0 = best_bc[i]
                # 8 neighbors + stay
                for db in [-step, 0, step]:
                    for dc in [-step, 0, step]:
                        if db == 0 and dc == 0:
                            continue
                        b_new = b0 + db
                        c_new = c0 + dc
                        if b_new < 0 or c_new < 0 or b_new + c_new > 1:
                            continue
                        trial_bc = best_bc.copy()
                        trial_bc[i] = [b_new, c_new]
                        trial_pts = _barycentric_to_cartesian(trial_bc)
                        trial_pts = _project_to_triangle(trial_pts)
                        trial_area = _min_triangle_area(trial_pts)
                        if trial_area > best_area:
                            best_area = trial_area
                            best_bc = trial_bc
                            improved = True
            if not improved:
                break
    
    return _barycentric_to_cartesian(best_bc), best_area


def _lem_heuristic(pts, max_iter=500, tol=1e-12):
    """Largest Empty Triangle heuristic: iteratively move the vertex of the
    minimum-area triangle toward the centroid of the convex hull of the
    remaining points to increase the minimum area."""
    pts = pts.copy()
    for it in range(max_iter):
        areas = _all_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        i, j, k = _TRIPLETS[min_idx]
        # Centroid of all points excluding i, j, k
        mask = np.ones(N, dtype=bool)
        mask[[i, j, k]] = False
        remaining = pts[mask]
        centroid = remaining.mean(axis=0)
        # For each of the three vertices, compute the new min area if we
        # move that vertex toward the centroid (parametric search)
        best_improve = 0.0
        best_move = None
        for v in [i, j, k]:
            # Try moving vertex v toward centroid
            # Parametrize: new_pos = pts[v] + t*(centroid - pts[v])
            # Find t in [0, 1] that maximizes the min area
            best_t = 0.0
            best_ma = min_area
            # Coarse search
            for t in np.linspace(0, 1, 21):
                trial = pts.copy()
                trial[v] = pts[v] + t * (centroid - pts[v])
                trial = _project_to_triangle(trial)
                ma = _min_triangle_area(trial)
                if ma > best_ma:
                    best_ma = ma
                    best_t = t
            # Fine search around best_t
            if best_t > 0:
                t_lo = max(0, best_t - 0.05)
                t_hi = min(1, best_t + 0.05)
                for t in np.linspace(t_lo, t_hi, 21):
                    trial = pts.copy()
                    trial[v] = pts[v] + t * (centroid - pts[v])
                    trial = _project_to_triangle(trial)
                    ma = _min_triangle_area(trial)
                    if ma > best_ma:
                        best_ma = ma
                        best_t = t
            improve = best_ma - min_area
            if improve > best_improve:
                best_improve = improve
                best_move = (v, best_t)
        if best_move is None or best_improve < tol:
            break
        v, t = best_move
        pts[v] = pts[v] + t * (centroid - pts[v])
        pts = _project_to_triangle(pts)
    return pts


def _gradient_step(pts, lr=0.005, max_iter=200):
    """Gradient ascent on soft-min area in barycentric coords."""
    bc = _cartesian_to_barycentric(pts)
    best_pts = pts.copy()
    best_area = _min_triangle_area(pts)
    for it in range(max_iter):
        areas = _all_areas(pts)
        beta = max(10.0, 50.0 / (best_area + 1e-8))
        grad_pts = _soft_min_grad(pts, beta)
        # Convert to barycentric gradient
        gb = grad_pts[:, 0]
        gc = 0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1]
        bc[:, 0] += lr * gb
        bc[:, 1] += lr * gc
        # Project to valid barycentric
        b = np.clip(bc[:, 0], 0, 1)
        c = np.clip(bc[:, 1], 0, 1 - b)
        bc = np.column_stack([b, c])
        pts = _barycentric_to_cartesian(bc)
        ma = _min_triangle_area(pts)
        if ma > best_area:
            best_area = ma
            best_pts = pts.copy()
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
    return best_pts


def _nm_polish(pts, max_iter=5000):
    bc = _cartesian_to_barycentric(pts)
    flat = bc.flatten()

    def obj(x):
        bc_x = x.reshape(-1, 2)
        bx = np.clip(bc_x[:, 0], 0, 1)
        cx = np.clip(bc_x[:, 1], 0, 1 - bx)
        px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
        return -_min_triangle_area(px)

    try:
        res = minimize(obj, flat, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-13, 'fatol': 1e-15})
        bc_res = res.x.reshape(-1, 2)
        b = np.clip(bc_res[:, 0], 0, 1)
        c = np.clip(bc_res[:, 1], 0, 1 - b)
        return _barycentric_to_cartesian(np.column_stack([b, c]))
    except Exception:
        return pts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        return np.array([u + 0.5 * v, v * S3H])

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1.0/3, S3H/3],
        [1.0/3, 0.0],
        [2.0/3, 0.0],
        [0.5, S3H/3],
        [0.5, 2*S3H/3],
    ], dtype=float)
    _update_best(init)

    # Phase 1: Discrete grid search (global)
    if _elapsed_ok(60):
        grid_pts, grid_area = _grid_search_discrete(n_starts=100, grid_size=30)
        if grid_pts is not None:
            _update_best(grid_pts)

    # Phase 2: Local refinement on grid solutions
    if best_pts is not None and _elapsed_ok(40):
        refined_pts, refined_area = _local_refinement_grid(best_pts, n_iter=50, grid_size=15)
        _update_best(refined_pts)

    # Phase 3: LEM heuristic for continuous refinement
    if best_pts is not None and _elapsed_ok(30):
        lem_pts = _lem_heuristic(best_pts, max_iter=300)
        _update_best(lem_pts)

    # Phase 4: Nelder-Mead polish
    if best_pts is not None and _elapsed_ok(15):
        p = _nm_polish(best_pts, max_iter=10000)
        _update_best(p)

    # Phase 5: Final fine polish
    if best_pts is not None and _elapsed_ok(5):
        p = _nm_polish(best_pts, max_iter=5000)
        _update_best(p)
        p = _lem_heuristic(p, max_iter=100)
        _update_best(p)

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    return _project_to_triangle(best_pts)
