"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 150
DEADLINE = 110.0  # seconds, safety margin from 120s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient wrt x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise differences: pts[i] - pts[j] for all i,j
    # Use broadcasting: shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = pts[i] - pts[j]
    dist = np.linalg.norm(diff, axis=2)  # shape (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -0.5 * sum_{i!=j} log(dist[i,j]) = -sum_{i<j} log(dist[i,j])
    # Using full matrix: sum over all i!=j is 2*sum over i<j
    # log(1/d) = -log(d)
    # energy = sum_{i<j} log(1/dist[i,j]) = -sum_{i<j} log(dist[i,j])
    # = -0.5 * sum_{i!=j} log(dist[i,j])
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient: for each i, grad_i = sum_{j != i} (pts[i] - pts[j]) / |pts[i] - pts[j]|^2
    # = sum_{j != i} diff[i,j] / dist[i,j]^2
    inv_dist_sq = np.zeros_like(dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    t_start = time.time()
    
    # Phase 1: Good initial configuration - Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit sphere
    pts = _project_to_sphere(pts)
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    elapsed = time.time() - t_start
    if elapsed > DEADLINE:
        return best_pts
    
    # Phase 2: Gradient-based optimization with adaptive step size
    # Use momentum-based optimization
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(pts)
    
    # First, do a few steps with larger learning rate to escape local minima
    for step in range(5000):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 5:  # Leave 5s for final polish
            break
        
        energy, grad = _energy_and_grad(pts)
        
        if not np.isfinite(energy):
            # Reset to best known
            pts = best_pts.copy()
            velocity = np.zeros_like(pts)
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Update with momentum
        velocity = momentum * velocity - lr * grad
        pts = pts + velocity
        pts = _project_to_sphere(pts)
        
        # Adaptive learning rate: reduce if energy increases
        if step % 100 == 99:
            if energy > best_energy * 1.01:  # if worse than best by 1%
                lr *= 0.8
            else:
                lr = min(lr * 1.05, 0.05)
    
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 2:
        return best_pts
    
    # Phase 3: Fine-tuning with smaller learning rate
    lr = 0.001
    momentum = 0.95
    velocity = np.zeros_like(pts)
    pts = best_pts.copy()
    
    for step in range(5000):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 1:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        if not np.isfinite(energy):
            pts = best_pts.copy()
            velocity = np.zeros_like(pts)
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        velocity = momentum * velocity - lr * grad
        pts = pts + velocity
        pts = _project_to_sphere(pts)
        
        # Very small adaptive adjustment
        if step % 200 == 199:
            if energy > best_energy:
                lr *= 0.9
            else:
                lr = min(lr * 1.02, 0.005)
    
    elapsed = time.time() - t_start
    if elapsed > DEADLINE:
        return best_pts
    
    # Phase 4: Final polish with very small steps
    lr = 0.0001
    momentum = 0.99
    velocity = np.zeros_like(pts)
    pts = best_pts.copy()
    
    for step in range(3000):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        if not np.isfinite(energy):
            break
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        velocity = momentum * velocity - lr * grad
        pts = pts + velocity
        pts = _project_to_sphere(pts)
        
        if step % 500 == 499:
            if energy > best_energy:
                lr *= 0.8
            else:
                lr = min(lr * 1.01, 0.0005)
    
    # Final validation
    pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Perturb slightly
        rng = np.random.default_rng(42)
        pts = pts + rng.normal(0, 1e-6, size=pts.shape)
        pts = _project_to_sphere(pts)
    
    return pts
