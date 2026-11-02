"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # High-resolution spherical grid for evaluating covering radius
    # Use a Fibonacci lattice with many more points for accurate radius estimation
    M = 5000  # number of grid points
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Normalize grid points (they should already be on unit sphere)
    grid_norms = np.linalg.norm(grid_points, axis=1, keepdims=True)
    grid_points = grid_points / grid_norms
    
    def covering_radius(points):
        """Compute the covering radius (max angular distance from grid to nearest point)."""
        # Compute squared geodesic distances from each grid point to each configuration point
        # Using dot product: cos(theta) = dot(p, q), theta = arccos(dot)
        # For small angles, theta ≈ sqrt(2*(1-dot))
        # We'll use the actual arccos for accuracy
        
        # Compute all pairwise dot products
        dots = grid_points @ points.T  # shape (M, N)
        # Clip to [-1, 1] for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        
        # Compute angular distances
        angles = np.arccos(dots)  # shape (M, N)
        
        # For each grid point, find the minimum angular distance
        min_angles = np.min(angles, axis=1)
        
        # Covering radius is the maximum of these minimums
        return np.max(min_angles)
    
    def objective_and_grad(x):
        """Objective function and its gradient using soft-min approximation."""
        points = x.reshape(-1, 3)
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Compute dot products
        dots = grid_points @ points.T  # shape (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        
        # Use log-sum-exp trick for smooth approximation of max
        # We want to approximate max_i f_i where f_i = -log(sum_j exp(-beta * d_ij^2))
        # Actually, let's use a different approach: smooth min then max
        
        # For each grid point, compute squared distances
        # d_ij^2 = 2*(1 - dot_ij) for unit vectors
        sq_dists = 2.0 * (1.0 - dots)  # shape (M, N)
        
        # Soft minimum over j for each i: approx min_j sq_dists[i,j]
        # Use -logsumexp(-beta * sq_dists) / beta
        beta = 50.0
        neg_beta_sq = -beta * sq_dists
        # Log-sum-exp for numerical stability
        max_vals = np.max(neg_beta_sq, axis=1, keepdims=True)
        logsumexp_vals = np.log(np.sum(np.exp(neg_beta_sq - max_vals), axis=1)) + max_vals.ravel()
        soft_min_sq = -logsumexp_vals / beta  # shape (M,)
        
        # Now we need max over i of soft_min_sq
        # Use another log-sum-exp for the max
        max_soft = np.max(soft_min_sq)
        logsumexp_max = np.log(np.sum(np.exp(soft_min_sq - max_soft))) + max_soft
        objective = logsumexp_max / beta  # approximate max of soft_min_sq
        
        # For gradient, we'll use finite differences if analytical is too complex
        # Or we can use the fact that L-BFGS-B can work with just function values
        return objective
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).ravel()
    
    best_x = x0.copy()
    best_val = covering_radius(x0.reshape(-1, 3))
    
    # Try multiple restarts with different initializations and optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the initial points
        if restart == 0:
            x_init = x0.copy()
        else:
            # Add random perturbation
            noise_scale = 0.1 * (restart / n_restarts)
            x_init = x0 + noise_scale * np.random.randn(len(x0))
        
        # Optimize using L-BFGS-B
        result = minimize(
            objective_and_grad,
            x_init,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000}
        )
        
        if time.time() > deadline:
            break
        
        x_opt = result.x
        # Normalize each point
        points_opt = x_opt.reshape(-1, 3)
        norms = np.linalg.norm(points_opt, axis=1, keepdims=True)
        points_opt = points_opt / norms
        
        val = covering_radius(points_opt)
        
        if val < best_val:
            best_val = val
            best_x = x_opt.copy()
    
    # Final refinement with Nelder-Mead if time permits
    if time.time() < deadline - 10.0:
        points_best = best_x.reshape(-1, 3)
        
        def nm_objective(x):
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            return covering_radius(points)
        
        result_nm = minimize(
            nm_objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if result_nm.fun < best_val:
            best_x = result_nm.x
            best_val = result_nm.fun
    
    # Final normalization and return
    points = best_x.reshape(-1, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
