import numpy as np
from scipy.optimize import minimize
import time

N = 130

def construct():
    start_time = time.time()
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0_flat = x0.flatten()
    
    # 2. Grid for Covering Radius Approximation
    # Use a dense Fibonacci spiral for the grid too
    N_grid = 50000
    i_grid = np.arange(N_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / N_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # 3. Objective Function
    beta = 50.0  # Softness parameter for log-sum-exp
    lambda_pen = 1000.0  # Penalty for leaving the sphere
    
    def objective(x_flat):
        # Time check
        if time.time() - start_time > 110:
            return 1e10  # Large value to stop optimization
        
        x = x_flat.reshape(N, 3)
        
        # Normalize for distance calculation
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        x_norm = x / norms
        
        # Compute squared distances from grid to all points
        # grid: (M, 3), x_norm: (N, 3)
        # dist_sq: (M, N)
        # dist_sq[j, i] = ||grid[j] - x_norm[i]||^2
        # = ||grid[j]||^2 - 2 grid[j] dot x_norm[i] + ||x_norm[i]||^2
        # Since both are on sphere, ||grid||=1, ||x_norm||=1
        # dist_sq = 2 - 2 * (grid @ x_norm.T)
        
        dots = grid @ x_norm.T  # (M, N)
        dist_sq = 2.0 - 2.0 * dots
        # Clip small negative values due to floating point errors
        dist_sq = np.maximum(dist_sq, 0.0)
        
        # Find min squared distance for each grid point
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Convert to angular distance (radians)
        # d = arccos(1 - d_sq/2)
        # For small angles, d ~ sqrt(d_sq)
        # Using sqrt is faster and smooth enough for optimization
        dists = np.sqrt(min_dist_sq)
        
        # Soft-max of distances
        # To avoid overflow, subtract max
        max_d = np.max(dists)
        if max_d == 0:
            soft_max_val = 0
        else:
            exp_terms = np.exp(beta * (dists - max_d))
            soft_max_val = (max_d + np.log(np.sum(exp_terms)) / beta)
            
        # Penalty for not being on the sphere
        pen = lambda_pen * np.sum((np.sum(x**2, axis=1) - 1.0)**2)
        
        return soft_max_val + pen

    # 4. Optimization
    # We will try to optimize. 
    # L-BFGS-B is good for this.
    
    best_x = x0_flat
    best_val = objective(best_x)
    
    try:
        res = minimize(
            objective, 
            best_x, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 1000}
        )
        
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception:
        pass
    
    # Final check and normalization
    final_x = best_x.reshape(N, 3)
    final_norms = np.linalg.norm(final_x, axis=1, keepdims=True)
    final_norms[final_norms == 0] = 1.0
    final_x = final_x / final_norms
    
    return final_x
