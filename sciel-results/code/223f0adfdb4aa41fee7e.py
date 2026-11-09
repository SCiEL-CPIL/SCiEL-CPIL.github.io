"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = 115.0  # seconds; safety margin before 120s kill
_PENALTY_LAMBDA = 1e6

def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _energy_and_grad(pts):
    """Compute energy and ambient gradient vectorized."""
    n = len(pts)
    # Compute all pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero
    dist_diag = np.diag_indices(n)
    dist[dist_diag] = 1.0
    
    # Check for close points
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        return 1e18, np.zeros_like(pts)
    
    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    E = -float(np.sum(np.log(d)))
    
    # Gradient calculation
    # grad[i] = - sum_{j != i} (pts[i] - pts[j]) / dist[i,j]^2
    factor = 1.0 / (dist * dist)
    factor[dist_diag] = 0.0
    
    sum_factor = np.sum(factor, axis=1, keepdims=True) # (N, 1)
    term1 = pts * sum_factor
    term2 = factor @ pts
    
    grad = -(term1 - term2)
    
    return E, grad

def _lbfgs_optimize(pts, max_iter=500, tol=1e-8):
    """L-BFGS-B optimization using penalty method."""
    pts = _project(pts.copy())
    
    def penalized_objective(x_flat):
        x = x_flat.reshape(N, 3)
        E, _ = _energy_and_grad(x)
        norms = np.linalg.norm(x, axis=1)
        penalty = _PENALTY_LAMBDA * np.sum((norms - 1.0)**2)
        return E + penalty
    
    def penalized_grad(x_flat):
        x = x_flat.reshape(N, 3)
        E, grad = _energy_and_grad(x)
        norms = np.linalg.norm(x, axis=1)
        penalty_grad = 2 * _PENALTY_LAMBDA * (norms - 1.0) * x / np.maximum(norms, 1e-15)
        return (grad + penalty_grad).flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(penalized_objective, x0, jac=penalized_grad, 
                         method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        opt_pts = _project(result.x.reshape(N, 3))
        opt_E, _ = _energy_and_grad(opt_pts)
        return opt_pts, opt_E
    except Exception:
        return pts, _energy_and_grad(pts)[0]

def _random_start(seed):
    """Generate a random set of points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _fibonacci_start():
    """Golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def construct():
    start_time = time.time()
    
    best_pts = _fibonacci_start()
    best_E, _ = _energy_and_grad(best_pts)
    
    # List of initializations
    initializations = [best_pts]
    
    # Add random starts
    for seed in range(50):
        if time.time() > _DEADLINE - 10:
            break
        initializations.append(_random_start(seed))
    
    # Optimize each initialization
    for init_pts in initializations:
        if time.time() > _DEADLINE - 5:
            break
        
        # Short optimization run
        opt_pts, E = _lbfgs_optimize(init_pts, max_iter=200, tol=1e-6)
        
        if E < best_E:
            best_E = E
            best_pts = opt_pts
    
    # Final polish on the best result
    if time.time() < _DEADLINE - 2:
        polished_pts, polished_E = _lbfgs_optimize(best_pts, max_iter=1000, tol=1e-9)
        if polished_E < best_E:
            best_E = polished_E
            best_pts = polished_pts
            
    return _project(best_pts)
# EVOLVE-BLOCK-END
