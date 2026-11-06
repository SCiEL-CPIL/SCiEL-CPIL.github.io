"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

def _fast_energy(points):
    """Compute logarithmic energy using vectorized NumPy operations.
    Returns (energy, min_dist_sq)"""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < 1e-12:
        return float('inf'), min_dist_sq
        
    # Energy = -0.25 * sum_{i!=j} log(dist_sq_ij)
    # Since sum_{i!=j} is 2 * sum_{i<j}, and log(1/d) = -0.5 log(d2)
    # E = sum_{i<j} -0.5 log(d2_ij) = -0.5 * 0.5 * sum_{i!=j} log(d2_ij) ? No.
    # sum_{i<j} log(1/d_ij) = sum_{i<j} -log(d_ij) = sum_{i<j} -0.5 log(d2_ij)
    # Let S = sum_{i!=j} log(d2_ij). Then sum_{i<j} log(d2_ij) = S/2.
    # E = -0.5 * (S/2) = -0.25 * S.
    energy = -0.25 * np.sum(np.log(dist_sq))
    return energy, min_dist_sq

def _fast_grad_tangent(points):
    """Compute gradient of log energy projected to tangent space. Vectorized."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    mask = ~np.eye(n, dtype=bool)
    if np.any(dist_sq[mask] < 1e-12):
        return None
        
    inv_d2 = 1.0 / dist_sq
    inv_d2[~mask] = 0.0
    
    weighted_diff = diff * inv_d2[:, :, np.newaxis]
    grad_raw = -np.sum(weighted_diff, axis=1)
    
    dot_products = np.sum(grad_raw * points, axis=1, keepdims=True)
    grad_tangent = grad_raw - dot_products * points
    
    return grad_tangent

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).astype(np.float64)

def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _gradient_descent(points, deadline, max_iters=1000, lr_init=0.05):
    """Run projected gradient descent with tangent-space gradient."""
    pts = points.copy()
    energy, _ = _fast_energy(pts)
    lr = lr_init
    lr_decay = 0.99
    lr_shrink = 0.7
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _fast_grad_tangent(pts)
        if grad is None:
            pts = pts + 0.01 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
            continue
        
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy, min_d2 = _fast_energy(new_pts)
        
        if new_energy < energy and min_d2 > 1e-10:
            pts = new_pts
            energy = new_energy
            lr *= lr_decay
        else:
            lr *= lr_shrink
            if lr < 1e-8:
                break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
    
    return pts, energy

def _polish(points, deadline):
    """Final polish with very small steps."""
    pts = points.copy()
    energy, _ = _fast_energy(pts)
    lr = 0.001
    
    for it in range(500):
        if time.time() > deadline:
            break
        grad = _fast_grad_tangent(pts)
        if grad is None:
            break
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy, min_d2 = _fast_energy(new_pts)
        if new_energy < energy and min_d2 > 1e-10:
            pts = new_pts
            energy = new_energy
            lr *= 0.98
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    return pts, energy

def construct():
    """Place N points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _fast_energy(best_pts)
    
    # Phase 1: GD from Fibonacci
    pts, energy = _gradient_descent(best_pts, deadline - 40.0, max_iters=1000, lr_init=0.05)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts.copy()
        
    # Phase 2: Random restarts
    np.random.seed(12345)
    for attempt in range(20):
        if time.time() > deadline - 20.0:
            break
        rand_pts = _random_sphere(N, seed=attempt * 7919 + 13)
        pts, energy = _gradient_descent(rand_pts, deadline - 10.0, max_iters=500, lr_init=0.02)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
    # Phase 3: Polish
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _polish(best_pts, deadline - 2.0)
        
    best_pts = _project_to_sphere(best_pts)
    
    # Final validity check
    _, min_d2 = _fast_energy(best_pts)
    if min_d2 < 1e-18 or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
        
    return best_pts
# EVOLVE-BLOCK-END
