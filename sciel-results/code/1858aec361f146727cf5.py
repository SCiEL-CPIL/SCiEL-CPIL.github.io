import numpy as np
import time

N = 150

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Riemannian Gradient Descent
    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # Grad_i = - sum_{j!=i} (X_i - X_j) / d_ij^2
    
    alpha = 0.01
    min_alpha = 1e-10
    max_iter = 5000
    start_time = time.time()
    time_limit = 140.0 # Leave 10s margin
    
    current_energy = np.inf
    # Calculate initial energy
    # Using NumPy for speed in optimization loop
    def calc_energy_grad(X):
        # X is (N, 3)
        # Compute pairwise distances
        # D[i, j] = ||X[i] - X[j]||
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        d = np.sqrt(d2)
        np.fill_diagonal(d, 1.0) # Avoid division by zero in log/grad calc for diagonal
        
        # Energy
        # We want sum over i<j of log(1/d)
        # log(1/d) = -log(d)
        # Mask upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = np.sum(-np.log(d[mask]))
        
        # Gradient
        # grad_i = - sum_{j!=i} (X_i - X_j) / d_ij^2
        # diff is (N, N, 3)
        # d2 is (N, N)
        # We need to divide diff by d2^2? No.
        # d = sqrt(d2). d^2 = d2.
        # term = (X_i - X_j) / d_ij^2
        # So we divide diff by d2[..., np.newaxis]
        # But we must avoid diagonal.
        # Set diagonal of d2 to 1 to avoid div by 0, then mask out later.
        d2_safe = d2.copy()
        np.fill_diagonal(d2_safe, 1.0)
        
        # grad = - sum_j (X_i - X_j) / d2_ij
        # diff[i, j] = X_i - X_j
        # We want sum over j for each i.
        # grad[i] = - sum_j diff[i, j] / d2_safe[i, j]
        
        # Multiply diff by 1/d2_safe
        inv_d2 = 1.0 / d2_safe
        # Zero out diagonal
        np.fill_diagonal(inv_d2, 0.0)
        
        # grad = - sum_j (diff * inv_d2[..., np.newaxis])
        grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        return energy, grad

    # Initial energy
    current_energy, grad = calc_energy_grad(X)
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Project gradient to tangent space
        # V = grad - (grad . X) X
        grad_dot_x = np.sum(grad * X, axis=1, keepdims=True)
        V = grad - grad_dot_x * X
        
        # Backtracking line search
        found = False
        for _ in range(20):
            X_new = X - alpha * V
            # Normalize
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            # Avoid zero norm
            norms[norms < 1e-12] = 1.0
            X_new = X_new / norms
            
            # Check validity (no points too close)
            # Simple check: min distance
            # This is O(N^2), but we already do it in calc_energy_grad
            new_energy, _ = calc_energy_grad(X_new)
            
            if new_energy < current_energy:
                X = X_new
                current_energy = new_energy
                found = True
                break
            else:
                alpha *= 0.5
        
        if not found:
            # If we couldn't find a step, maybe reduce alpha further or break
            # If alpha is too small, we are at a local minimum or stuck
            if alpha < min_alpha:
                break
            # Try a smaller alpha and continue
            alpha *= 0.5
            continue
        
        # Adaptive alpha increase?
        # If we succeeded with a relatively large step, we can try to increase it slightly
        # But be careful.
        # Let's keep it simple: constant or slowly decaying alpha.
        # Or just keep the last successful alpha.
        
    # Final polish with scipy.optimize.minimize if time permits?
    # RGD should be good enough.
    # Ensure output is normalized
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    return X
