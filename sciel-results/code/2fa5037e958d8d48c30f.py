"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math


def _energy_and_gradient(pts):
    """Compute logarithmic energy and its gradient on the sphere.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    """
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum over i<j of -log(sqrt(dist_sq_ij)) = -0.5 * sum_{i!=j} log(sqrt(dist_sq_ij))
    # = -0.25 * sum_{i!=j} log(dist_sq_ij)
    # But we only want i<j, so:
    # E = sum_{i<j} log(1/sqrt(dist_sq_ij)) = -0.5 * sum_{i<j} log(dist_sq_ij)
    
    # For gradient: dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # Note: (x_k - x_j) / |x_k - x_j|^2 = diff[k,j] / dist_sq[k,j]
    
    # Gradient computation
    # grad[k] = sum_{j != k} diff[k, j] / dist_sq[k, j]
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Energy
    # Use upper triangle for i<j
    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[i_idx, j_idx])
    # Avoid log(0)
    dists = np.maximum(dists, 1e-300)
    E = -np.sum(np.log(dists))
    
    return E, grad


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: good initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure they're on the unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Gradient descent with momentum and adaptive step size
    lr = 0.01
    momentum = 0.9
    vel = np.zeros_like(pts)
    
    # Run optimization
    best_pts = pts.copy()
    best_E, _ = _energy_and_gradient(pts)
    
    # Use adaptive learning rate with line search-like behavior
    max_iters = 5000
    
    for it in range(max_iters):
        E, grad = _energy_and_gradient(pts)
        
        # Update velocity with momentum
        vel = momentum * vel - lr * grad
        
        # Take step
        new_pts = pts + vel
        
        # Project back to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        # Check if energy improved
        new_E, _ = _energy_and_gradient(new_pts)
        
        if new_E < E:
            pts = new_pts
            if new_E < best_E:
                best_E = new_E
                best_pts = pts.copy()
            # Decay learning rate slowly
            lr *= 0.999
        else:
            # Reduce learning rate and retry
            lr *= 0.5
            vel = momentum * vel
            new_pts = pts + vel
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / norms
            new_E, _ = _energy_and_gradient(new_pts)
            if new_E < E:
                pts = new_pts
                if new_E < best_E:
                    best_E = new_E
                    best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
