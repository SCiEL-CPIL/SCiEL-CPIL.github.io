import numpy as np
import time
import math

N = 150

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Momentum
    v = np.zeros_like(x)
    momentum = 0.9
    
    # Learning rate schedule
    lr = 1e-4
    lr_decay = 0.999
    
    best_energy = float('inf')
    best_x = x.copy()
    
    # Precompute indices
    idx = np.arange(N)
    
    while time.time() - start_time < time_limit:
        # Compute pairwise differences
        # x shape (N, 3)
        # diff[i, j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        
        # Distances
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero on diagonal
        dist_sq[idx, idx] = 1.0
        dist = np.sqrt(dist_sq)
        
        # Gradient of log(1/d) = -log(d) wrt x_i
        # d/dx_i (-log(|x_i - x_j|)) = -(x_i - x_j) / |x_i - x_j|^2
        # Wait, energy is sum log(1/d) = - sum log(d).
        # Gradient of -log(d) is - (1/d) * (x_i - x_j)/d = -(x_i - x_j)/d^2.
        # We want to minimize energy, so we move in direction of -gradient.
        # -grad = (x_i - x_j)/d^2.
        
        # G[i, j] = (x_i - x_j) / dist_sq[i, j]
        # We need to sum over j for each i.
        # diff is (N, N, 3). dist_sq is (N, N).
        # Factor = 1 / dist_sq.
        factor = 1.0 / dist_sq
        factor[idx, idx] = 0.0
        
        # grad[i] = sum_j factor[i, j] * (x[i] - x[j])
        # This is equivalent to (diff * factor[:, :, np.newaxis]).sum(axis=1)
        grad = np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        # proj_grad = grad - (grad . x) x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        proj_grad = grad - dot * x
        
        # Update velocity
        v = momentum * v - lr * proj_grad
        
        # Update position
        x = x + v
        
        # Normalize
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        
        # Compute energy for monitoring (approximate, for tracking best)
        # E = - sum_{i<j} log(dist)
        # Use upper triangle
        triu_dist = dist[np.triu_indices(N, k=1)]
        # Filter out very small distances if any, though shouldn't happen
        triu_dist = triu_dist[triu_dist > 1e-12]
        if len(triu_dist) > 0:
            energy = -np.sum(np.log(triu_dist))
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
        
        # Decay learning rate
        lr *= lr_decay
        
        # If lr is too small, break
        if lr < 1e-10:
            break
            
    return best_x
