"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    n = len(pts)
    # Compute all pairwise distances
    # Use broadcasting: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log of very small numbers
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())

def _energy_grad(pts):
    """Compute gradient of logarithmic energy.
    
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = -sum_{j!=i} (xi - xj) / |xi-xj|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff**2, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-30)
    # grad_i = -sum_j (xi - xj) / |xi-xj|^2
    # = -sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    """Project points back onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n, offset=0.0):
    """Generate Fibonacci sphere points with optional offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize_on_sphere(pts_init, max_iter=5000, tol=1e-10):
    """Optimize points on sphere using L-BFGS-B with re-projection."""
    n = len(pts_init)
    # Flatten and work in 3N-dimensional space with projection
    
    def _energy_flat(x):
        pts = x.reshape(n, 3)
        return _energy(pts)
    
    def _grad_flat(x):
        pts = x.reshape(n, 3)
        return _energy_grad(pts).flatten()
    
    # Use L-BFGS-B
    result = minimize(
        _energy_flat,
        pts_init.flatten(),
        jac=_grad_flat,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    pts_opt = result.x.reshape(n, 3)
    pts_opt = _project_to_sphere(pts_opt)
    return pts_opt, result

def _local_search(pts, n_iter=1000, lr_init=0.01):
    """Simple gradient descent with re-projection as a fast local optimizer."""
    n = len(pts)
    pts = pts.copy()
    lr = lr_init
    
    for it in range(n_iter):
        grad = _energy_grad(pts)
        pts = pts - lr * grad
        pts = _project_to_sphere(pts)
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_pts = None
    best_energy = float('inf')
    
    def _update_best(pts):
        nonlocal best_pts, best_energy
        try:
            e = _energy(pts)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Generate multiple initial configurations
    inits = []
    
    # Fibonacci sphere with various offsets
    for offset in [0.0, 0.1, 0.25, 0.5, 0.75, 0.3, 0.6]:
        inits.append(_fibonacci_sphere(N, offset))
    
    # Random inits
    for seed in [42, 123, 456, 789, 1000, 2024]:
        inits.append(_random_sphere(N, seed))
    
    # Sort by initial energy
    inits_with_energy = []
    for pts in inits:
        e = _energy(pts)
        inits_with_energy.append((e, pts))
    inits_with_energy.sort(key=lambda x: x[0])
    
    # Try optimizing each initial configuration
    for idx, (init_e, pts_init) in enumerate(inits_with_energy):
        if time.time() > deadline - 20.0:
            break
        
        # Quick local search first
        try:
            pts_ls = _local_search(pts_init, n_iter=500, lr_init=0.005)
            _update_best(pts_ls)
        except Exception:
            pass
        
        if time.time() > deadline - 15.0:
            break
        
        # Then L-BFGS-B for more precise optimization
        try:
            pts_lbfgs, result = _optimize_on_sphere(pts_ls, max_iter=2000, tol=1e-12)
            _update_best(pts_lbfgs)
        except Exception:
            pass
        
        # If we still have time, do a final polish
        if time.time() > deadline - 5.0:
            break
        
        if best_pts is not None and idx >= 3:
            # Refine the best found so far
            try:
                pts_polish = _local_search(best_pts, n_iter=1000, lr_init=0.001)
                _update_best(pts_polish)
            except Exception:
                pass
    
    # Final polish pass if time permits
    if best_pts is not None and time.time() < deadline - 3.0:
        try:
            pts_final = _local_search(best_pts, n_iter=2000, lr_init=0.0005)
            _update_best(pts_final)
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return Fibonacci sphere
        best_pts = _fibonacci_sphere(N)
    
    # Ensure all points are on unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
