import numpy as np
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Precompute indices for pairs
    # For gradient calculation, we can do it in a loop for clarity and simplicity, 
    # as N=100 is small enough (100*100 = 10k ops per step).
    
    def energy_and_grad(X):
        # X is (N, 3)
        # Compute pairwise distances
        # d_ij = ||x_i - x_j||
        # Using broadcasting
        X1 = X[:, np.newaxis, :]  # (N, 1, 3)
        X2 = X[np.newaxis, :, :]  # (1, N, 3)
        diff = X1 - X2  # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=2)  # (N, N)
        
        # Avoid division by zero (diagonal)
        # We only care about i != j
        np.fill_diagonal(dist_sq, 1.0) # Prevent div by 0 for grad calculation mask
        
        dist = np.sqrt(dist_sq)
        
        # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
        # log(1/d) = -log(d)
        # Sum over i<j is half the sum over i!=j
        log_d = np.log(dist)
        np.fill_diagonal(log_d, 0.0)
        E = -0.5 * np.sum(log_d)
        
        # Gradient of E w.r.t x_i
        # grad_i = sum_{j!=i} (x_j - x_i) / dist_ij^2
        # Note: dist_sq is (N, N)
        # term_ij = (x_j - x_i) / dist_sq_ij
        # grad_i = sum_j term_ij
        
        # diff is x_i - x_j. We want x_j - x_i, so -diff.
        # term = -diff / dist_sq[:, :, np.newaxis]
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Calculate gradient contribution from each pair
        # grad[i] = sum_j (x_j - x_i) / dist_sq[i, j]
        # Let's compute a matrix G where G[i, j] is the vector (x_j - x_i) / dist_sq[i, j]
        # Then grad[i] = sum_j G[i, j]
        
        # Avoid division by zero on diagonal by masking
        safe_dist_sq = np.where(mask, dist_sq, 1.0)
        
        # term_vectors[i, j, k] = (x_j[i, j, k] - x_i[i, j, k]) / safe_dist_sq[i, j]
        # diff[i, j, k] = x_i[k] - x_j[k]
        # So x_j - x_i = -diff
        term_vectors = -diff / safe_dist_sq[:, :, np.newaxis]
        
        # Zero out diagonal
        term_vectors[~mask] = 0.0
        
        # Sum over j (axis=1)
        grad = np.sum(term_vectors, axis=1)
        
        return E, grad

    # Gradient Descent on Sphere
    eta = 0.05
    best_E, best_X = energy_and_grad(X)
    best_E, best_X = best_E, X.copy()
    
    while time.time() < deadline:
        E, G = energy_and_grad(X)
        
        # Project gradient to tangent space
        # g_t = g - (g . x) x
        dot = np.sum(G * X, axis=1, keepdims=True)
        G_t = G - dot * X
        
        # Adaptive step size
        # Try step
        X_new = X - eta * G_t
        X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
        
        E_new, _ = energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            # Increase step slightly if it worked well? Keep stable.
            # eta = min(eta * 1.05, 0.5) 
        else:
            # Reduce step
            eta = eta * 0.5
            
            if eta < 1e-6:
                # Try a random restart or break
                # For now, break to avoid infinite loop of tiny steps
                # We could also try a random perturbation
                break

    # Final check
    E_final, _ = energy_and_grad(X)
    if E_final < best_E:
        best_X = X
    else:
        # Use best found
        pass
        
    return best_X
