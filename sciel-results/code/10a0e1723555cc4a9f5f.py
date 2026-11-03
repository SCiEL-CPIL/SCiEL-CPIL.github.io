"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return total

def _compute_energy_fast(pts):
    """Vectorized energy computation"""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)  # Avoid log(0)
    # Only sum upper triangle
    iu = np.triu_indices(n, k=1)
    return math.fsum(math.log(1.0 / d) for d in dist[iu])

def _compute_gradient(pts):
    """Compute gradient of log energy w.r.t. each point in R^3"""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                continue
            # d/dx_i [log(1/d)] = -(x_i - x_j)/d^2
            grad[i] += -d_vec / (d * d)
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere"""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere"""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_from_start(pts, max_iter=2000, lr_init=0.01, deadline=None):
    """Gradient descent with projection to sphere"""
    pts = _project_to_sphere(pts.copy())
    current_energy = _compute_energy_fast(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    lr = lr_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _compute_gradient(pts)
        
        # Project gradient to tangent space
        for i in range(len(pts)):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Take gradient step
        pts_new = pts - lr * grad
        pts_new = _project_to_sphere(pts_new)
        
        new_energy = _compute_energy_fast(pts_new)
        
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
            lr = min(lr * 1.05, 0.1)
        else:
            lr = lr * 0.5
            if lr < 1e-8:
                break
    
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy"""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_spiral(N)
    
    # Optimize from Fibonacci start
    best_pts, best_energy = _optimize_from_start(pts0, max_iter=3000, lr_init=0.005, deadline=deadline)
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random start with some structure
        if restart == 0:
            # Slightly perturbed Fibonacci
            pts_start = pts0 + rng.normal(0, 0.05, size=(N, 3))
        else:
            # Random points
            pts_start = rng.normal(size=(N, 3))
        
        pts_opt, energy_opt = _optimize_from_start(pts_start, max_iter=1500, lr_init=0.005, deadline=deadline)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Final polish with very small steps if time permits
    if time.time() < deadline - 2.0:
        for _ in range(500):
            if time.time() > deadline - 1.0:
                break
            grad = _compute_gradient(best_pts)
            for i in range(N):
                grad[i] -= np.dot(grad[i], best_pts[i]) * best_pts[i]
            best_pts = best_pts - 0.0001 * grad
            best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END

