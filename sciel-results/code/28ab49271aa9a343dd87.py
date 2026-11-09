"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute pairwise distances
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Sum log(1/d) = -sum(log(d)) over i<j
    # Take upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log of very small numbers
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _energy_and_grad(points):
    """Compute energy and gradient of logarithmic energy on sphere.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Gradient with respect to x_k (in R^3):
    grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    
    But we need the gradient projected onto the tangent space of the sphere.
    The Riemannian gradient is (I - x_k x_k^T) grad_k = grad_k - (x_k · grad_k) x_k
    """
    n = len(points)
    grad = np.zeros_like(points)
    
    # Compute all pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    
    # dist_sq[i,j] = |x_i - x_j|^2
    # grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # In matrix form: grad_k = sum_j diff[k,j] / dist_sq[k,j]
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot = np.sum(points * grad, axis=1)
    grad = grad - dot[:, np.newaxis] * points
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _optimize(points, time_limit, seed=0):
    """Optimize points on sphere using gradient descent with adaptive step size."""
    rng = np.random.RandomState(seed)
    points = points.copy()
    best_points = points.copy()
    best_energy = _energy(points)
    
    start_time = time.time()
    
    # Adaptive step size
    lr = 0.01
    prev_energy = best_energy
    
    # Use L-BFGS-like approach with line search
    max_iter = 100000
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(points)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Backtracking line search
        alpha = lr
        max_alpha = 0.1
        min_alpha = 1e-8
        
        # Ensure we stay on sphere: use exponential map
        # x_new = normalize(x + alpha * grad)
        
        for _ in range(50):
            # Move in direction of negative gradient (we want to minimize energy)
            # But gradient points in direction of increasing energy
            # So we move in direction -grad
            new_points = _normalize(points - alpha * grad)
            
            new_energy = _energy(new_points)
            
            if new_energy < energy:
                break
            alpha *= 0.5
            if alpha < min_alpha:
                break
        
        else:
            # No improvement found, try smaller step
            alpha *= 0.5
            new_points = _normalize(points - alpha * grad)
            new_energy = _energy(new_points)
        
        if new_energy < energy:
            points = new_points
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points.copy()
            
            # Adaptive learning rate
            if new_energy < energy - 1e-6:
                lr = min(lr * 1.1, 0.1)
            else:
                lr = max(lr * 0.9, 1e-5)
        else:
            lr = max(lr * 0.5, 1e-6)
            if lr < 1e-8:
                break
    
    return best_points, best_energy

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline (10s safety margin)
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # Phase 1: Gradient descent from Fibonacci start
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        points, energy = _optimize(points, time_remaining - 5.0, seed=42)
    
    # Phase 2: Try multiple random restarts if time permits
    best_points = points
    best_energy = energy
    
    time_remaining = deadline - time.time()
    if time_remaining > 10.0:
        rng = np.random.RandomState(123)
        n_restarts = min(5, int(time_remaining // 10.0))
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random perturbation of best points
            noise_scale = 0.1 * (1.0 - restart / max(n_restarts, 1))
            perturbed = best_points + rng.normal(0, noise_scale, size=best_points.shape)
            perturbed = _normalize(perturbed)
            
            t_rem = deadline - time.time() - 2.0
            if t_rem < 1.0:
                break
            
            pts, e = _optimize(perturbed, t_rem, seed=100 + restart)
            
            if e < best_energy:
                best_energy = e
                best_points = pts
    
    # Final check: ensure points are on unit sphere
    best_points = _normalize(best_points)
    
    # Verify no points are too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci sphere
        best_points = _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
