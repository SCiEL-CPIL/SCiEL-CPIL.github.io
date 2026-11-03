import numpy as np
import time

N = 150

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Optimization: Projected Gradient Descent with Momentum
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Gradient wrt x_i: -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    
    # Hyperparameters
    momentum = 0.9
    step_size = 0.01
    min_step_size = 1e-6
    decay_rate = 0.999
    
    V = np.zeros_like(X) # Velocity for momentum
    
    while time.time() < deadline:
        # Compute pairwise differences and distances
        # X shape (N, 3)
        # Diff shape (N, N, 3)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero for i==j
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Compute Gradient
        # G_i = -sum_{j!=i} (x_i - x_j) / dist_ij^2
        # Note: diff[i,j] = x_i - x_j
        # We need sum over j for each i
        # inv_dist_sq = 1 / dist_sq
        # G = -sum_j (diff[i,j] / dist_sq[i,j])
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0) # Exclude self-interaction
        
        # G[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
        # diff shape (N, N, 3), inv_dist_sq shape (N, N)
        # Multiply diff by inv_dist_sq[..., np.newaxis]
        weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
        G = -np.sum(weighted_diff, axis=1)
        
        # Update with Momentum
        V = momentum * V - step_size * G
        X = X + V
        
        # Project to Unit Sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        # Decay step size
        step_size *= decay_rate
        if step_size < min_step_size:
            break
            
    # Final check and normalization
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    return X
