import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = ~np.eye(n, dtype=bool)
    
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / np.maximum(dist_sq[mask], 1e-30)
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral():
    i = np.arange(N, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)

def _perturbed_fibonacci(seed=42, scale=0.1):
    rng = np.random.RandomState(seed)
    base = _fibonacci_spiral()
    noise = rng.randn(N, 3) * scale
    return _project_to_sphere(base + noise)

def _optimize_lbfgs(pts_init, maxiter=3000, ftol=1e-12, gtol=1e-10, time_limit=30.0):
    start = time.time()
    C_pen = 1e10
    x0 = pts_init.flatten().copy()
    
    def objective(x):
        pts = x.reshape(N, 3)
        pts_sph = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts_sph)
        norms_sq = np.sum(pts * pts, axis=1)
        penalty = C_pen * np.sum((norms_sq - 1.0) ** 2)
        return e + penalty
    
    def jac(x):
        pts = x.reshape(N, 3)
        pts_sph = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts_sph)
        norms_sq = np.sum(pts * pts, axis=1)
        g_pen = 4.0 * C_pen * pts * (norms_sq - 1.0)[:, np.newaxis]
        return (g + g_pen).flatten()
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'maxfun': maxiter * 2})
        if time.time() - start < time_limit:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass
    
    result = _project_to_sphere(best_x.reshape(N, 3))
    return result, best_val

def _polish_ultra(pts, maxiter=10000, ftol=1e-15, gtol=1e-14, time_limit=30.0):
    start = time.time()
    C_pen = 1e12
    x0 = pts.flatten().copy()
    
    def objective(x):
        pts = x.reshape(N, 3)
        pts_sph = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts_sph)
        norms_sq = np.sum(pts * pts, axis=1)
        penalty = C_pen * np.sum((norms_sq - 1.0) ** 2)
        return e + penalty
    
    def jac(x):
        pts = x.reshape(N, 3)
        pts_sph = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts_sph)
        norms_sq = np.sum(pts * pts, axis=1)
        g_pen = 4.0 * C_pen * pts * (norms_sq - 1.0)[:, np.newaxis]
        return (g + g_pen).flatten()
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'maxfun': maxiter * 3})
        if time.time() - start < time_limit:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass
    
    result = _project_to_sphere(best_x.reshape(N, 3))
    return result, best_val

def _basin_hopping(pts_init, n_iter=10, step_scale=0.1, time_limit=60.0):
    start = time.time()
    best_pts = _project_to_sphere(pts_init.copy())
    best_val = _energy_and_grad(best_pts)[0]
    rng = np.random.RandomState(42)
    
    for i in range(n_iter):
        if time.time() - start > time_limit:
            break
        
        # Perturb in tangent space
        noise = rng.randn(N, 3) * step_scale
        perturbed = _project_to_sphere(best_pts + noise)
        
        # Local optimization
        opt_pts, val = _optimize_lbfgs(perturbed, maxiter=2000, ftol=1e-12, gtol=1e-10, time_limit=5.0)
        
        if val < best_val:
            best_val = val
            best_pts = opt_pts
            step_scale = max(0.01, step_scale * 0.8)
        else:
            step_scale = min(0.5, step_scale * 1.2)
    
    return best_pts, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial: Fibonacci spiral
    pts_fib = _fibonacci_spiral()
    
    # First optimization
    best_pts, best_val = _optimize_lbfgs(pts_fib, maxiter=3000, ftol=1e-12, gtol=1e-10, time_limit=20.0)
    
    # Basin hopping from best
    if time.time() < deadline - 50.0:
        bh_pts, bh_val = _basin_hopping(best_pts, n_iter=15, step_scale=0.15, time_limit=40.0)
        if bh_val < best_val:
            best_val = bh_val
            best_pts = bh_pts
    
    # Try diverse initializations
    inits = [
        _random_points(42),
        _random_points(123),
        _perturbed_fibonacci(456, 0.2),
        _perturbed_fibonacci(789, 0.3),
    ]
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        opt_pts, val = _optimize_lbfgs(init, maxiter=2000, ftol=1e-12, gtol=1e-10, time_limit=10.0)
        if val < best_val:
            best_val = val
            best_pts = opt_pts
    
    # Final ultra-tight polish
    if time.time() < deadline - 5.0:
        best_pts, best_val = _polish_ultra(best_pts, maxiter=10000, ftol=1e-15, gtol=1e-14, time_limit=20.0)
    
    best_pts = _project_to_sphere(best_pts)
    return best_pts
