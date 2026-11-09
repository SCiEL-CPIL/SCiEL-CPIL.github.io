import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _raw_log_energy(pts):
    """Compute the exact logarithmic energy using math.fsum for accuracy."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())

def _energy_and_grad(x_flat, lambda_pen=100.0):
    """Compute energy and gradient for L-BFGS-B optimization."""
    x = x_flat.reshape(N, 3)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    log_dists = np.log(dist)
    log_dists[~mask] = 0.0
    E_log = -0.5 * np.sum(log_dists)
    
    norms_sq = np.sum(x * x, axis=1)
    pen = lambda_pen * np.sum((norms_sq - 1.0)**2)
    
    total_energy = E_log + pen
    
    grad_log = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * x
    total_grad = grad_log + grad_pen
    
    return total_energy, total_grad.flatten()

def _project_to_sphere(pts):
    """Project points exactly onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_points(n=N, offset=0.0):
    """Generate Fibonacci spiral points on the unit sphere with optional offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)

def _random_sphere_points(n=N, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _optimize_from_start(x0, t_start, time_limit):
    """Run L-BFGS-B optimization from a starting point, returning best result."""
    elapsed = time.time() - t_start
    if elapsed > time_limit - 5:
        return None, None
    
    try:
        res = minimize(
            _energy_and_grad, 
            x0.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': 300,
                'ftol': 1e-10,
                'gtol': 1e-6,
                'maxls': 50
            }
        )
        
        candidate = _project_to_sphere(res.x.reshape(N, 3))
        cand_energy = _raw_log_energy(candidate)
        return candidate, cand_energy
    except Exception:
        return None, None

def _polish(x, t_start, time_limit, max_iter=500, ftol=1e-14, gtol=1e-10):
    """Polish a configuration with tighter tolerances."""
    elapsed = time.time() - t_start
    if elapsed > time_limit - 5:
        return x
    
    try:
        res = minimize(
            _energy_and_grad, 
            x.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 100
            }
        )
        
        candidate = _project_to_sphere(res.x.reshape(N, 3))
        return candidate
    except Exception:
        return x

def construct():
    t_start = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Generate diverse starting points
    starts = []
    
    # Fibonacci points with different offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.5]:
        starts.append(_fibonacci_points(N, offset=offset))
    
    # Random points with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        starts.append(_random_sphere_points(N, seed=seed))
    
    best_x = starts[0].copy()
    best_energy = _raw_log_energy(best_x)
    
    # Try each starting point with L-BFGS-B
    for i, x0 in enumerate(starts):
        elapsed = time.time() - t_start
        if elapsed > time_limit - 30:
            break
        
        candidate, cand_energy = _optimize_from_start(x0, t_start, time_limit)
        
        if candidate is not None and cand_energy < best_energy:
            best_x = candidate
            best_energy = cand_energy
    
    # Polish the best result
    elapsed = time.time() - t_start
    if elapsed < time_limit - 20:
        best_x = _polish(best_x, t_start, time_limit, max_iter=500, ftol=1e-14, gtol=1e-10)
        best_energy = _raw_log_energy(best_x)
    
    # Final high-precision polish if time remains
    elapsed = time.time() - t_start
    if elapsed < time_limit - 10:
        best_x = _polish(best_x, t_start, time_limit, max_iter=300, ftol=1e-16, gtol=1e-12)
    
    # Final normalization
    best_x = _project_to_sphere(best_x)
    
    return best_x
