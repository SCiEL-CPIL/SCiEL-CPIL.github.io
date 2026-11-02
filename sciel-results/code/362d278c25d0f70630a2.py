"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _compute_energy(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    n = len(pts)
    dots = np.dot(pts, pts.T)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    log_inv_dist = -np.log(dist)
    np.fill_diagonal(log_inv_dist, 0.0)
    triu_indices = np.triu_indices(n, k=1)
    energy = math.fsum(log_inv_dist[triu_indices].tolist())
    return energy

def _project_to_sphere(pts):
    """Normalize each point to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _vectorized_gradient(pts):
    """
    Compute the gradient of the logarithmic energy on the sphere.
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    dE/dxi = -sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = pts.shape[0]
    # Pairwise differences: shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Squared distances: shape (n, n)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Inverse squared distances, mask diagonal
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    # Avoid division by zero
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    inv_dist_sq[mask] = 1.0 / dist_sq_safe[mask]
    
    # Weighted differences: shape (n, n, 3)
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    # This sum is -dE/dxi
    grad_neg = np.sum(weighted_diff, axis=1)
    
    # True gradient is -grad_neg
    grad = -grad_neg
    
    # Project gradient to tangent space
    # grad_tan_i = grad_i - (grad_i . x_i) x_i
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_proj = grad - grad_dot_x * pts
    
    return grad_proj

def _optimize_from_start(start_pts, max_iters=2000, lr_init=0.1, momentum=0.9):
    """
    Optimize points on sphere using vectorized gradient descent with momentum.
    """
    pts = _project_to_sphere(start_pts.copy())
    best_energy = _compute_energy(pts)
    best_pts = pts.copy()
    
    v = np.zeros_like(pts)
    lr = lr_init
    
    for i in range(max_iters):
        grad = _vectorized_gradient(pts)
        
        # Momentum update
        # v = momentum * v - lr * grad
        v = momentum * v - lr * grad
        new_pts = pts + v
        
        # Project to sphere
        new_pts = _project_to_sphere(new_pts)
        
        energy = _compute_energy(new_pts)
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If energy increased or invalid, reduce learning rate
            lr *= 0.5
            
            # If lr is too small, stop
            if lr < 1e-6:
                break
                
        # Decay learning rate slowly
        if i % 100 == 0:
            lr *= 0.95

    return best_pts, best_energy

def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    time_limit = 115.0  # Leave some buffer
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib_pts)
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        rand_pts = rng.normal(size=(N, 3))
        configs.append(_project_to_sphere(rand_pts))
    
    best_pts_overall = None
    best_energy_overall = float('inf')
    
    for idx, start_pts in enumerate(configs):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Run optimization
        try:
            optimized_pts, energy = _optimize_from_start(start_pts, max_iters=5000, lr_init=0.05, momentum=0.9)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_pts_overall = optimized_pts
        except Exception as e:
            pass
            
    # If we still have time, try to polish the best result found
    if best_pts_overall is not None and time.time() - start_time < time_limit - 5:
        # Run a finer optimization on the best result
        try:
            polished_pts, energy = _optimize_from_start(best_pts_overall, max_iters=5000, lr_init=0.01, momentum=0.99)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_pts_overall = polished_pts
        except Exception:
            pass

    # Final validation and return
    if best_pts_overall is None:
        return _project_to_sphere(fib_pts)
        
    return _project_to_sphere(best_pts_overall)

# EVOLVE-BLOCK-END
