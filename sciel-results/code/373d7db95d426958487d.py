import numpy as np
from math import fsum

N = 150

# EVOLVE-BLOCK-START

def construct():
    # 1. Initialize with Fibonacci spiral (good starting point)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Gradient Descent Optimization
    # We minimize E = sum log(1/d) = -sum log(d)
    # Gradient w.r.t x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # This is the repulsive force. We move in the direction of this force to increase distances, lowering energy.
    
    # Hyperparameters
    # Start with a moderate step size and decay it.
    # N=150 is small enough for O(N^2) operations in Python/NumPy to be fast within 120s.
    num_steps = 5000
    initial_lr = 0.05
    
    for step in range(num_steps):
        # Calculate forces
        # dist_sq[i, j] = ||x_i - x_j||^2
        # We need to avoid division by zero. Points are distinct initially and we normalize, but numerical issues can arise.
        
        # Vectorized distance calculation
        # x shape (N, 3)
        # diff[i, j, k] = x[i, k] - x[j, k]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Set diagonal to infinity to avoid self-interaction division by zero
        np.fill_diagonal(dist_sq, np.inf)
        
        # Force term: (x_i - x_j) / dist_sq[i, j]
        # diff / dist_sq[:,:,np.newaxis]
        # Where dist_sq is inf, result is 0.
        inv_dist_sq = 1.0 / dist_sq
        inv_dist_sq[~np.isfinite(inv_dist_sq)] = 0.0
        
        # Force on i from j: (x_i - x_j) / ||x_i - x_j||^2
        # diff[i,j] is x_i - x_j
        # So force_vec = diff * inv_dist_sq[:,:,np.newaxis]
        force = diff * inv_dist_sq[:, :, np.newaxis]
        
        # Sum over j to get total force on each i
        # force shape (N, N, 3)
        # Sum over axis 1 (j)
        total_force = np.sum(force, axis=1)
        
        # Project force onto tangent plane
        # tangent_force = total_force - dot(total_force, x) * x
        dot_products = np.sum(total_force * x, axis=1)
        tangent_force = total_force - dot_products[:, np.newaxis] * x
        
        # Update position
        # Adaptive learning rate: decay over time to allow convergence
        # lr = initial_lr * (1.0 - step / num_steps)
        # Or exponential decay
        lr = initial_lr * (0.95 ** (step / 50.0))
        
        x = x + lr * tangent_force
        
        # Normalize to keep on unit sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        # Avoid division by zero if a point collapses (unlikely with repulsion)
        norms[norms == 0] = 1.0
        x = x / norms
        
        # Optional: Check for early stopping if energy change is tiny
        if step > 1000 and step % 100 == 0:
             # Calculate energy roughly to check convergence (expensive, so do it sparsely)
             pass
             
    # Final check for validity
    if not np.all(np.isfinite(x)):
        # Fallback to Fibonacci if something went wrong
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return x

# EVOLVE-BLOCK-END
