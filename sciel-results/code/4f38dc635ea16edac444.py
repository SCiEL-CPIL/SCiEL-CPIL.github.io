"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy_and_grad(pts):
    """Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|) and its gradient."""
    n = pts.shape[0]
    # Pairwise distances
    # dist[i,j] = |xi - xj|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j of log(dist[i,j])
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Because d/dx_i log(1/|xi-xj|) = -d/dx_i log|xi-xj| = -(xi-xj)/|xi-xj|^2
    # Wait: E = sum_{i<j} -log|xi-xj|. 
    # dE/dx_k = sum_{j: k<j} - (x_k - x_j)/|x_k - x_j|^2 + sum_{i: i<k} - (x_j - x_k)/|x_j - x_k|^2
    # Note: (x_j - x_k)/|x_j - x_k|^2 = -(x_k - x_j)/|x_k - x_j|^2
    # So dE/dx_k = - sum_{j != k} (x_k - x_j)/|x_k - x_j|^2
    
    # Let's compute grad properly.
    # grad[i] = - sum_{j != i} (pts[i] - pts[j]) / dist[i,j]^2
    # diff[i,j] = pts[i] - pts[j]
    # dist2[i,j] = dist[i,j]^2
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)
    
    # We want G[i] = - sum_{j != i} diff[i,j] / dist2[i,j]
    # diff is (n, n, 3). dist2 is (n, n).
    # We can do: G = - sum(axis=1) of (diff / dist2[:, :, np.newaxis])
    # But diagonal is 0/1 = 0, so it's safe if we handle diagonal correctly.
    # Actually diff[i,i] = 0, so 0/1 = 0.
    inv_dist2 = 1.0 / dist2
    inv_dist2[np.diag_indices(n)] = 0.0  # Ensure diagonal is 0
    
    # diff * inv_dist2[:, :, np.newaxis]
    # Sum over axis 1 (j)
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _retract(pts, grad, step):
    """Retract gradient step onto the sphere."""
    # Move in direction of negative gradient
    new_pts = pts - step * grad
    # Project back to sphere
    return _project_to_sphere(new_pts)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute initial energy
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    # Gradient descent with adaptive step size
    step = 0.01
    min_step = 1e-6
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Try a step
        new_pts = _retract(pts, grad, step)
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            # Increase step slightly if improvement is good
            if new_energy < energy - 1e-8:
                step = min(step * 1.1, 0.1)
        else:
            # Decrease step
            step *= 0.8
            if step < min_step:
                break
        
        # Convergence check
        if np.linalg.norm(grad) < 1e-10:
            break
    
    # Final polish with smaller steps
    step = 1e-5
    for it in range(500):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(best_pts)
        new_pts = _retract(best_pts, grad, step)
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            best_pts = new_pts
            best_energy = new_energy
    
    return best_pts
# EVOLVE-BLOCK-END
