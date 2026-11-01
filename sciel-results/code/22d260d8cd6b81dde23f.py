"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time


def construct():
    # Vectorized energy and gradient computation
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize to unit sphere (handle zero norms just in case)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = x / norms
        
        # Compute pairwise squared distances
        # d_ij^2 = ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi.xj = 2 - 2 xi.xj
        # Using dot product is faster and more stable for small distances? 
        # Actually, for log energy, we need d_ij.
        # d_ij = sqrt(2 - 2 * dot)
        
        dots = np.dot(x, x.T)
        # Clip to avoid negative due to floating point errors
        dots = np.clip(dots, -1.0, 1.0)
        dist_sq = 2.0 - 2.0 * dots
        # Avoid log(0) for i=j
        np.fill_diagonal(dist_sq, 1.0)
        
        dist = np.sqrt(dist_sq)
        
        # Energy = -sum_{i<j} log(dist_ij)
        iu = np.triu_indices(N, k=1)
        log_dists = np.log(dist[iu])
        energy = -np.sum(log_dists)
        
        # Gradient
        # E = -sum_{i<j} log(||xi - xj||)
        # grad_k = -sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
        # Note: ||x_k - x_j||^2 = dist_sq[k,j]
        
        # We need sum_j (x_k - x_j) / dist_sq[k,j]
        # Let A[k,j] = 1 / dist_sq[k,j]
        # Then grad_k = sum_j A[k,j] * (x_k - x_j)
        #              = x_k * sum_j A[k,j] - sum_j A[k,j] * x_j
        
        A = 1.0 / dist_sq
        # Set diagonal to 0 to exclude self-interaction in sum
        np.fill_diagonal(A, 0.0)
        
        # sum_j A[k,j] is the row sum of A
        row_sums = np.sum(A, axis=1)
        
        # sum_j A[k,j] * x_j is A @ x
        Ax = A @ x
        
        # grad_k = x_k * row_sums[k] - Ax[k]
        grad = x * row_sums[:, np.newaxis] - Ax
        
        # Project gradient onto tangent space
        # grad_tangent = grad - (grad . x_k) x_k
        dots_grad = np.sum(grad * x, axis=1)
        grad = grad - dots_grad[:, np.newaxis] * x
        
        return energy, grad.flatten()

    best_energy = float('inf')
    best_x = None
    
    # Number of restarts. 
    # A single run of L-BFGS-B on 150 points takes ~1-5 seconds depending on convergence.
    # We have 120 seconds. Let's try 10-20 restarts.
    # Random initializations are better for exploring different basins.
    n_restarts = 10
    
    rng = np.random.default_rng(42)
    
    for k in range(n_restarts):
        # Random initialization on sphere
        x0 = rng.normal(size=(N, 3))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        
        try:
            res = minimize(
                lambda x: energy_and_grad(x)[0],
                x0.flatten(),
                jac=lambda x: energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000}
            )
            
            x_opt = res.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            x_opt = x_opt / norms
            
            # Check validity
            if not np.all(np.isfinite(x_opt)):
                continue
                
            # Calculate final energy
            e, _ = energy_and_grad(x_opt.flatten())
            
            if e < best_energy:
                best_energy = e
                best_x = x_opt
                
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to Fibonacci if all fail
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_x
# EVOLVE-BLOCK-END
