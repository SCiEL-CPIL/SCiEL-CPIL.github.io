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

def _energy_and_grad(x_flat):
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
    
    lambda_pen = 10.0
    norms_sq = np.sum(x * x, axis=1)
    pen = lambda_pen * np.sum((norms_sq - 1.0)**2)
    
    total_energy = E_log + pen
    
    grad_log = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * x
    total_grad = grad_log + grad_pen
    
    return total_energy, total_grad.flatten()

def _energy_and_grad_strict(x_flat):
    """Compute energy and gradient for high-precision polish with strict constraint."""
    x = x_flat.reshape(N, 3)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    log_dists = np.log(dist)
    log_dists[~mask] = 0.0
    E_log = -0.5 * np.sum(log_dists)
    
    lambda_pen = 100.0
    norms_sq = np.sum(x * x, axis=1)
    pen = lambda_pen * np.sum((norms_sq - 1.0)**2)
    
    total_energy = E_log + pen
    
    grad_log = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * x
    total_grad = grad_log + grad_pen
    
    return total_energy, total_grad.flatten()

def _fibonacci_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    return x0

def _random_sphere_points(seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def construct():
    t_start = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    best_x = None
    best_energy = float('inf')
    
    # Try multiple initializations
    initializations = [_fibonacci_points()]
    
    # Add more random restarts
    for seed in [42, 123, 456, 789, 1011, 1213]:
        initializations.append(_random_sphere_points(seed))
    
    for init_idx, x0 in enumerate(initializations):
        if time.time() - t_start > time_limit - 30:
            break
        
        try:
            res = minimize(
                _energy_and_grad, 
                x0.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            x_candidate = res.x.reshape(N, 3)
            # Normalize to sphere
            x_candidate = x_candidate / np.linalg.norm(x_candidate, axis=1, keepdims=True)
            
            # Compute raw energy (without penalty) for comparison
            e_raw = _raw_log_energy(x_candidate)
            
            if e_raw < best_energy:
                best_energy = e_raw
                best_x = x_candidate.copy()
                
        except Exception:
            continue
    
    # High-precision polish pass if time permits
    if best_x is not None and time.time() - t_start < time_limit - 10:
        try:
            # Use strict penalty for final polish to ensure points are on sphere
            res2 = minimize(
                _energy_and_grad_strict, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 200
                }
            )
            
            x_polished = res2.x.reshape(N, 3)
            x_polished = x_polished / np.linalg.norm(x_polished, axis=1, keepdims=True)
            
            e_polished = _raw_log_energy(x_polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_x = x_polished.copy()
                
        except Exception:
            pass
    
    # Final normalization and validation
    if best_x is not None:
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
        # Ensure no points are too close
        diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, 1.0)
        if np.min(dist) < 1e-9:
            # Fallback to Fibonacci if points are too close
            best_x = _fibonacci_points()
    
    return best_x
