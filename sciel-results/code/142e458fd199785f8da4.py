import numpy as np
import scipy.optimize as opt
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds buffer
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x0 = X.flatten()
    
    def energy_and_grad_flat(x):
        X = x.reshape(N, 3)
        # Compute pairwise distances
        # d[i, j] = ||X[i] - X[j]||
        # Use broadcasting
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dist = np.maximum(dist, 1e-12)
        
        # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
        # Create upper triangle mask
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = -np.sum(np.log(dist[mask]))
        
        # Gradient: grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
        # dist_sq[i, j] = d_ij^2
        dist_sq = dist ** 2
        # Gradient term for each pair (i, j) is (x_j - x_i) / d_ij^2
        # We need to sum over j for each i.
        # grad[i] = sum_j (X[j] - X[i]) / dist_sq[i, j]
        # Note: dist_sq has 0 on diagonal. We should ignore diagonal.
        # Set diagonal to 1 to avoid division by zero, then mask it out.
        np.fill_diagonal(dist_sq, 1.0)
        
        # grad_contrib[i, j] = (X[j] - X[i]) / dist_sq[i, j]
        # This is a (N, N, 3) tensor.
        # grad_contrib[i, j, k] = (X[j, k] - X[i, k]) / dist_sq[i, j]
        
        # Efficient calculation:
        # We want grad[i] = sum_j (X[j] - X[i]) / dist_sq[i, j]
        # = sum_j X[j] / dist_sq[i, j] - X[i] * sum_j 1 / dist_sq[i, j]
        
        # Let W[i, j] = 1 / dist_sq[i, j] for i != j, 0 otherwise.
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        # Sum over j:
        # term1[i, k] = sum_j X[j, k] * W[i, j]
        term1 = W @ X  # (N, 3)
        
        # term2[i, k] = X[i, k] * sum_j W[i, j]
        row_sums = W.sum(axis=1) # (N,)
        term2 = X * row_sums[:, np.newaxis]
        
        grad = term1 - term2
        
        # Gradient of energy is -grad?
        # E = - sum log(d).
        # dE/dx_i = - sum_j (1/d) * (x_i - x_j)/d = - sum_j (x_i - x_j)/d^2 = sum_j (x_j - x_i)/d^2.
        # So grad is correct as derived.
        
        # Penalty term
        # P = mu * sum_i (||x_i||^2 - 1)^2
        # dP/dx_i = mu * 2 (||x_i||^2 - 1) * 2 x_i = 4 mu x_i (||x_i||^2 - 1)
        mu = 1000.0
        norms_sq = np.sum(X**2, axis=1)
        penalty_grad = 4.0 * mu * X * (norms_sq - 1.0)[:, np.newaxis]
        
        total_grad = grad + penalty_grad
        
        # Penalty value
        penalty_val = mu * np.sum((norms_sq - 1.0)**2)
        total_energy = energy + penalty_val
        
        return total_energy, total_grad.flatten()

    # Optimization
    try:
        res = opt.minimize(
            energy_and_grad_flat, 
            x0, 
            jac=True, 
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
        )
        
        if time.time() > deadline:
            return X # Return initial if timeout (shouldn't happen with good start)
            
        X_opt = res.x.reshape(N, 3)
        
        # Normalize to ensure they are exactly on the sphere
        norms = np.linalg.norm(X_opt, axis=1)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        X_opt = X_opt / norms[:, np.newaxis]
        
        return X_opt

    except Exception:
        return X

# EVOLVE-BLOCK-END
