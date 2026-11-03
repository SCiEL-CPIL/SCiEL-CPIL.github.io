import numpy as np
import time

N = 100

def energy(points):
    """Calculate the logarithmic energy."""
    # points: (N, 3)
    # Compute pairwise distances
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid log(0) on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    # Sum log(1/d) = -sum log(d)
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))

def gradient(points):
    """Calculate the gradient of the energy with respect to points."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # term_ij = diff[i, j] / dist_sq[i, j]
    # We need to sum over j for each i
    # diff is (N, N, 3), dist_sq is (N, N)
    # We want to divide diff by dist_sq[..., None]
    # But we must exclude j=i
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # To avoid division by zero, we can use where
    # Or simply set diagonal of dist_sq to 1 and multiply by mask later
    inv_dist_sq = 1.0 / dist_sq
    inv_dist_sq[~mask] = 0.0
    
    grad_terms = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(grad_terms, axis=1)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial best: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_energy = energy(best_points)
    
    rng = np.random.default_rng(42)
    
    num_restarts = 50 # Max number of restarts, will break based on time
    
    for k in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize points
        # Strategy: Fibonacci + noise, or random
        if k == 0:
            # Already have best_points from Fibonacci, but let's try a slight perturbation
            x = best_points + rng.normal(0, 0.01, (N, 3))
        else:
            # Random initialization
            x = rng.normal(0, 1, (N, 3))
            x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
            
        # Normalize initial points
        x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
        
        # Gradient Descent
        lr = 0.01
        min_lr = 1e-6
        max_iters = 1000
        
        for it in range(max_iters):
            if time.time() > deadline - 1.0:
                break
                
            grad = gradient(x)
            
            # Project gradient to tangent space
            proj_grad = grad - np.sum(grad * x, axis=1)[:, np.newaxis] * x
            
            # Adaptive learning rate: scale by norm of gradient?
            # Simple decay
            x_new = x - lr * proj_grad
            x_new = x_new / np.linalg.norm(x_new, axis=1)[:, np.newaxis]
            
            # Check for convergence or improvement
            # Calculate energy
            e = energy(x_new)
            
            if e < best_energy:
                best_energy = e
                best_points = x_new.copy()
                
            # Update x
            x = x_new
            
            # Decay learning rate
            lr *= 0.995
            if lr < min_lr:
                break
                
    return best_points
