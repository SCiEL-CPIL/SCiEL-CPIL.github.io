"""Optimized program: L-BFGS-B with polish pass for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 15.0  # seconds before the deadline to stop and return best


def _log_energy(pts):
    """Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    dist = np.sqrt(np.maximum(dist2, 1e-30))
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient.
    
    E = -sum_{i<j} log(d_ij)
    grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    dist = np.sqrt(np.maximum(dist2, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist2 = np.zeros_like(dist2)
        mask = dist2 > 1e-30
        inv_dist2[mask] = 1.0 / dist2[mask]
    
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _normalize(pts):
    """Normalize each point to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangents):
    """Move points on sphere using exponential map: x_new = cos(t) * x + sin(t) * t_hat"""
    pts = np.asarray(pts, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    
    # Project tangents to tangent space
    dots = np.sum(pts * tangents, axis=1, keepdims=True)
    tangents = tangents - dots * pts
    
    # Compute step sizes
    step_sizes = np.linalg.norm(tangents, axis=1, keepdims=True)
    step_sizes = np.maximum(step_sizes, 1e-15)
    
    # Exponential map
    cos_t = np.cos(step_sizes)
    sin_t = np.sin(step_sizes)
    t_hat = tangents / step_sizes
    
    new_pts = cos_t * pts + sin_t * t_hat
    return _normalize(new_pts)


def _halton_sphere(n, base1=2, base2=3):
    """Generate n points on the unit sphere using a 2D Halton sequence."""
    def halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        u = halton(i, base1)
        v = halton(i, base2)
        
        theta = math.acos(1.0 - 2.0 * v)
        phi = 2.0 * math.pi * u
        
        points[i-1, 0] = math.sin(theta) * math.cos(phi)
        points[i-1, 1] = math.sin(theta) * math.sin(phi)
        points[i-1, 2] = math.cos(theta)
    
    return _normalize(points)


def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _lbfgs_optimize(pts, max_iter=2000, gtol=1e-8, time_budget=30.0):
    """Run L-BFGS-B optimization on the sphere using tangent space projection."""
    start_time = time.time()
    
    def objective(x_flat):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts_local = _normalize(x_flat.reshape(N, 3))
        e, _ = _energy_and_grad(pts_local)
        return e
    
    def grad_func(x_flat):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts_local = _normalize(x_flat.reshape(N, 3))
        _, g = _energy_and_grad(pts_local)
        # Project to tangent space
        dots = np.sum(g * pts_local, axis=1)
        g_tangent = g - dots[:, np.newaxis] * pts_local
        return g_tangent.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': gtol, 'maxfun': 5000, 'disp': False}
        )
        
        if result.fun < float('inf'):
            pts_opt = _normalize(result.x.reshape(N, 3))
            return pts_opt, result.fun
    except Exception:
        pass
    
    return pts, float('inf')


def _polish(pts, max_iter=5000, gtol=1e-10, time_budget=20.0):
    """High-precision polish pass with very tight tolerances."""
    start_time = time.time()
    
    def objective(x_flat):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts_local = _normalize(x_flat.reshape(N, 3))
        e, _ = _energy_and_grad(pts_local)
        return e
    
    def grad_func(x_flat):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts_local = _normalize(x_flat.reshape(N, 3))
        _, g = _energy_and_grad(pts_local)
        dots = np.sum(g * pts_local, axis=1)
        g_tangent = g - dots[:, np.newaxis] * pts_local
        return g_tangent.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': gtol, 'maxfun': 10000, 'disp': False}
        )
        
        if result.fun < float('inf'):
            pts_opt = _normalize(result.x.reshape(N, 3))
            return pts_opt, result.fun
    except Exception:
        pass
    
    return pts, float('inf')


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate multiple initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Halton sequence
    initializations.append(_halton_sphere(N))
    
    # Random with different seeds
    for seed in [42, 123, 789]:
        np.random.seed(seed)
        pts_rand = np.random.randn(N, 3)
        initializations.append(_normalize(pts_rand))
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Evaluate initial energy
        e0, _ = _energy_and_grad(init_pts)
        
        if e0 < best_energy:
            best_energy = e0
            best_pts = init_pts.copy()
        
        # Run L-BFGS-B optimization
        remaining_time = max(deadline - time.time(), 5.0)
        alloc_time = min(remaining_time / (len(initializations) - init_idx), 25.0)
        
        try:
            pts_opt, e_opt = _lbfgs_optimize(
                init_pts, 
                max_iter=2000, 
                gtol=1e-8,
                time_budget=alloc_time
            )
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
        
        # If we have time, do a polish pass on the best result
        if time.time() < deadline and best_pts is not None:
            remaining_time = max(deadline - time.time(), 5.0)
            if remaining_time > 10.0:
                try:
                    pts_polished, e_polished = _polish(
                        best_pts,
                        max_iter=5000,
                        gtol=1e-10,
                        time_budget=min(remaining_time, 20.0)
                    )
                    
                    if e_polished < best_energy:
                        best_energy = e_polished
                        best_pts = pts_polished.copy()
                except Exception:
                    pass
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
    
    # Final projection and validation
    best_pts = _normalize(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                np.random.seed(999)
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts[j] = _normalize(best_pts[j])
    
    return best_pts
