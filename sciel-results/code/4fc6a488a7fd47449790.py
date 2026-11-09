import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Precompute indices for pairs
    # We need to calculate gradient for each point k: sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    # This can be done vectorized.
    
    best_energy = float('inf')
    best_x = x.copy()
    
    # Adaptive step size
    eta = 0.01
    eta_min = 1e-6
    decay = 0.999
    
    # Gradient Descent Loop
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Calculate pairwise distances
        # diff[i, j] = x_i - x_j
        # dist[i, j] = ||x_i - x_j||
        # To avoid division by zero, we add a small epsilon or handle it.
        # Since points are distinct initially, and we normalize, distances shouldn't be 0 unless we crash.
        
        # Vectorized gradient calculation
        # grad_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
        # Let D be the matrix of squared distances. D[i, j] = ||x_i - x_j||^2
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # D = np.sum(diff**2, axis=2)
        # np.fill_diagonal(D, 1.0) # Avoid div by 0 on diagonal
        # inv_D = 1.0 / D
        # np.fill_diagonal(inv_D, 0.0)
        # grad = np.sum(diff * inv_D[:, :, np.newaxis], axis=1)
        
        # This is O(N^2) memory and compute. N=150 is small (150^2 = 22500), so this is fine.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        D = np.sum(diff**2, axis=2)
        # Ensure no division by zero for identical points (shouldn't happen, but safe)
        D[np.diag_indices(N)] = 1.0
        inv_D = 1.0 / D
        np.fill_diagonal(inv_D, 0.0)
        
        # Gradient of energy E = sum log(1/d) = -sum log(d)
        # dE/dx_k = sum_j -1/d * d(d)/dx_k = sum_j -1/d * (x_k - x_j)/d = -sum_j (x_k - x_j)/d^2
        # We want to minimize, so update x = x - eta * grad
        # x_new = x - eta * (-sum (x_k - x_j)/d^2) = x + eta * sum (x_k - x_j)/d^2
        # Wait, let's re-verify signs.
        # f(d) = log(1/d) = -log(d).
        # df/dx_k = -1/d * (x_k - x_j)/d = -(x_k - x_j)/d^2.
        # Gradient descent: x <- x - eta * grad.
        # x <- x - eta * ( - sum (x_k - x_j)/d^2 ) = x + eta * sum (x_k - x_j)/d^2.
        # This term sum (x_k - x_j)/d^2 points AWAY from other points?
        # Let x_k = (1,0), x_j = (0,0). x_k - x_j = (1,0). d^2 = 1.
        # Term is (1,0). x_k moves to (1+eta, 0). This increases distance.
        # Energy log(1/d) decreases as d increases. So we want to increase d.
        # So moving away is correct for minimization of log(1/d).
        
        grad = np.sum(diff * inv_D[:, :, np.newaxis], axis=1)
        
        # Update
        x_new = x + eta * grad
        
        # Project back to sphere
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        x_new = x_new / norms
        
        # Calculate energy to check improvement
        # E = sum_{i<j} log(1/d_ij)
        # We already have D (squared distances). d = sqrt(D).
        # We need sum over i<j of -log(sqrt(D_ij)) = -0.5 * sum_{i<j} log(D_ij)
        # Let's compute it efficiently.
        # D is upper/lower triangular symmetric.
        # mask = np.triu(np.ones((N,N), dtype=bool), 1)
        # E = -0.5 * np.sum(np.log(D[mask]))
        # This is O(N^2) per step.
        
        mask = np.triu(np.ones((N, N), dtype=bool), 1)
        current_energy = -0.5 * np.sum(np.log(D[mask]))
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_x = x_new.copy()
            
        # Decay step size
        eta *= decay
        if eta < eta_min:
            # If step size is too small, we might be stuck or converged.
            # We can break or keep going with minimal steps.
            # Given time limit, breaking is safer if no progress is made recently.
            # But let's just continue with min eta for a bit longer to ensure local min.
            pass
            
        x = x_new
        
    return best_x
