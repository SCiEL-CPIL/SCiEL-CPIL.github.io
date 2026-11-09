import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def construct():
    # Start time
    t_start = time.time()
    time_limit = 120.0 # Safety margin
    
    # 1. Initial Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure exact unit length
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)

    best_x = x0
    best_energy = float('inf')

    def energy_and_grad(x_flat):
        # Reshape to (N, 3)
        x = x_flat.reshape(N, 3)
        
        # Calculate pairwise distances
        # x[:, np.newaxis, :] - x[np.newaxis, :, :] gives (N, N, 3)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero (though points should be distinct)
        # Set diagonal to 1 to avoid log(0) or div by 0
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Log energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # We can compute sum_{i!=j} log(dist) and divide by 2, but be careful with diagonal.
        # Let's compute the full matrix sum of -log(dist) for i!=j, then divide by 2.
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Energy term
        log_dists = np.log(dist)
        log_dists[~mask] = 0.0
        E_log = -0.5 * np.sum(log_dists)
        
        # Penalty term: lambda * sum(||x_i||^2 - 1)^2
        # We use a small lambda to keep points on sphere
        lambda_pen = 10.0
        norms_sq = np.sum(x * x, axis=1)
        pen = lambda_pen * np.sum((norms_sq - 1.0)**2)
        
        total_energy = E_log + pen
        
        # Gradient
        # Grad of E_log: 
        # d/dx_i (-0.5 * sum_{j!=i} log(d_ij))
        # = -0.5 * sum_{j!=i} (1/d_ij) * d(d_ij)/dx_i
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # So term is -0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
        
        # We can vectorize this.
        # diff is (N, N, 3) where diff[i,j] = x_i - x_j
        # dist_sq is (N, N)
        # We want sum over j for each i of -0.5 * diff[i,j] / dist_sq[i,j]
        
        # Set diagonal of dist_sq to 1 to avoid div by 0 in gradient calc too
        # diff diagonal is 0, so 0/1 = 0, which is correct.
        
        grad_log = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        
        # Grad of Penalty:
        # d/dx_i (lambda * (||x_i||^2 - 1)^2) = lambda * 2 * (||x_i||^2 - 1) * 2 * x_i
        grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * x
        
        total_grad = grad_log + grad_pen
        
        return total_energy, total_grad.flatten()

    # Run L-BFGS-B
    try:
        res = minimize(
            energy_and_grad, 
            x0.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        
        if res.success or res.fun < best_energy:
            best_x = res.x.reshape(N, 3)
            # Normalize best_x
            best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
            # Recalculate energy on normalized points to be sure
            # The energy function above includes penalty, so we need the raw log energy for comparison
            # But the optimizer minimized E+Pen. If Pen is small, it's close.
            # Let's just use the position.
            
    except Exception as e:
        pass

    # Check time
    if time.time() - t_start < time_limit - 5:
        # Do a second pass or refinement if time permits?
        # L-BFGS-B is usually good. Maybe a steepest descent polish?
        # For now, return the result.
        pass

    # Final normalization
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x
