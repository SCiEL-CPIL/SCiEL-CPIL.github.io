"""Heilbronn 11 triangles: Barycentric soft-min with DE global search and SA polish."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_to_triangle(pts):
    """Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _penalized_objective(flat_params):
    """Objective for DE: negative min area + penalty for out-of-triangle points."""
    points = flat_params.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 1000
        if y > S3H:
            penalty += (y - S3H) * 1000
        if x < 0:
            penalty += abs(x) * 1000
        if x > 1:
            penalty += (x - 1) * 1000
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 1000
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 1000
    ma = _min_triangle_area(points)
    return -ma + penalty


def _barycentric_objective(flat_params, beta):
    """Objective in barycentric space: negative soft-min area.
    Uses (u, v) barycentric coordinates where u>=0, v>=0, u+v<=1.
    Point = u*V0 + v*V1 + (1-u-v)*V2 with V0=(0,0), V1=(1,0), V2=(0.5,S3H).
    x = v + 0.5*(1-u-v) = 0.5 - 0.5u + 0.5v
    y = S3H*(1-u-v)
    """
    bary = flat_params.reshape(-1, 2)
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    s = u + v
    mask = s > 1
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    x = 0.5 - 0.5 * u + 0.5 * v
    y = S3H * (1.0 - u - v)
    pts = np.column_stack([x, y])
    return -_soft_min_area(pts, beta)


def _barycentric_hard_objective(flat_params):
    """Hard objective in barycentric space for Nelder-Mead."""
    bary = flat_params.reshape(-1, 2)
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    s = u + v
    mask = s > 1
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    x = 0.5 - 0.5 * u + 0.5 * v
    y = S3H * (1.0 - u - v)
    pts = np.column_stack([x, y])
    return -_min_triangle_area(pts)


def _cart_to_uv(pts):
    """Convert Cartesian to (u,v) barycentric coordinates.
    y = S3H*(1-u-v) => u+v = 1 - y/S3H
    x = 0.5 - 0.5u + 0.5v => v - u = 2x - 1
    v = (2x - 1 + 1 - y/S3H)/2 = x - y/(2*S3H)
    u = 1 - y/S3H - v = 1 - y/S3H - x + y/(2*S3H) = 1 - x - y/(2*S3H)
    """
    x = pts[:, 0]
    y = pts[:, 1]
    u = 1.0 - x - y / (2.0 * S3H)
    v = x - y / (2.0 * S3H)
    return np.column_stack([u, v])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [1/6, S3H/6],
        [5/6, S3H/6],
        [1/3, 2*S3H/3],
        [2/3, 2*S3H/3],
    ], dtype=float)
    _update_best(init)

    # ---- Phase A: Differential Evolution for global exploration ----
    if _elapsed_ok(60):
        try:
            bounds = [(0, 1), (0, S3H)] * N
            de_result = differential_evolution(
                _penalized_objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            de_pts = de_result.x.reshape(-1, 2)
            de_pts = _project_to_triangle(de_pts)
            _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Multi-start barycentric soft-min optimization ----
    # Build diverse starting points in (u,v) barycentric space
    init_uv = _cart_to_uv(init)
    bases_uv = [init_uv.copy()]
    if best_pts is not None:
        bases_uv.append(_cart_to_uv(best_pts))
    
    for _ in range(8):
        noise = rng.normal(0, 0.2, size=(N, 2))
        b = init_uv + noise
        b = np.clip(b, 0, 1)
        s = b[:, 0] + b[:, 1]
        mask = s > 1
        if np.any(mask):
            b[mask, 0] /= s[mask]
            b[mask, 1] /= s[mask]
        bases_uv.append(b)

    beta_schedule = [15.0, 40.0, 100.0, 300.0, 800.0, 2000.0]
    n_starts = 12
    
    for s in range(n_starts):
        if not _elapsed_ok(35):
            break
        base = bases_uv[s % len(bases_uv)]
        uv = base + rng.normal(0, 0.05, size=(N, 2))
        uv = np.clip(uv, 0, 1)
        ss = uv[:, 0] + uv[:, 1]
        mask = ss > 1
        if np.any(mask):
            uv[mask, 0] /= ss[mask]
            uv[mask, 1] /= ss[mask]

        for beta in beta_schedule:
            if not _elapsed_ok(30):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _barycentric_objective(x, b),
                    uv.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                uv = res.x.reshape(-1, 2)
                u = np.clip(uv[:, 0], 0, 1)
                v = np.clip(uv[:, 1], 0, 1)
                x = 0.5 - 0.5 * u + 0.5 * v
                y = S3H * (1.0 - u - v)
                pts = np.column_stack([x, y])
                pts = _project_to_triangle(pts)
                _update_best(pts)
            except Exception:
                pass

    # ---- Phase C: Simulated Annealing polish in Cartesian space ----
    if best_pts is not None and _elapsed_ok(30):
        current_pts = best_pts.copy()
        current_area = _min_triangle_area(current_pts)
        best_sa_pts = current_pts.copy()
        best_sa_area = current_area
        
        T_start = 0.02
        T_end = 0.0005
        n_iter = 5000
        
        for i in range(n_iter):
            if not _elapsed_ok(5):
                break
            T = T_start * (T_end / T_start) ** (i / n_iter)
            
            perturbation = rng.normal(0, 0.015, size=(N, 2))
            new_pts = current_pts + perturbation
            new_pts = _project_to_triangle(new_pts)
            new_area = _min_triangle_area(new_pts)
            
            if new_area >= current_area:
                current_pts = new_pts
                current_area = new_area
                if new_area > best_sa_area:
                    best_sa_area = new_area
                    best_sa_pts = new_pts.copy()
            else:
                delta = new_area - current_area
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_area = new_area
        
        _update_best(best_sa_pts)

    # ---- Phase D: Nelder-Mead local polish in barycentric space ----
    if best_pts is not None and _elapsed_ok(15):
        try:
            uv_init = _cart_to_uv(best_pts)
            res = minimize(
                _barycentric_hard_objective,
                uv_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            uv = res.x.reshape(-1, 2)
            u = np.clip(uv[:, 0], 0, 1)
            v = np.clip(uv[:, 1], 0, 1)
            x = 0.5 - 0.5 * u + 0.5 * v
            y = S3H * (1.0 - u - v)
            pts = np.column_stack([x, y])
            pts = _project_to_triangle(pts)
            _update_best(pts)
        except Exception:
            pass

    # ---- Phase E: Final fine polish with Nelder-Mead in Cartesian ----
    if best_pts is not None and _elapsed_ok(5):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                nm_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        pts = init.copy()
        best_pts = pts

    return best_pts
