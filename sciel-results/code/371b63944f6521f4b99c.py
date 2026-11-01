import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Gradient Descent
    step_size = 1e-2
    momentum = 0.9
    V = np.zeros_like(X)
    
    for k in range(10000):
        if time.time() - start_time > 140: # Safety margin
            break
            
        # Compute pairwise differences
        Diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D2 = np.sum(Diff**2, axis=2)
        np.fill_diagonal(D2, 1.0) # Avoid div by 0
        InvD2 = 1.0 / D2
        np.fill_diagonal(InvD2, 0.0)
        
        # Negative Gradient of Energy (Repulsive Force)
        # F_i = sum_j (x_i - x_j) / D_ij^2
        F = np.sum(Diff * InvD2[:, :, np.newaxis], axis=1)
        
        # Project to tangent space
        Dot = np.sum(F * X, axis=1)
        Tangent = F - Dot[:, np.newaxis] * X
        
        # Update with momentum (Move along Force)
        V = momentum * V + step_size * Tangent
        X = X + V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        
        # Decay step size
        if k > 500:
            step_size *= 0.999
            
    return X
# EVOLVE-BLOCK-END
