"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    t_start = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Start with hexagonal lattice as initial guess
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            initial_points = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
    
    # Compute pairwise distances for optimization
    def compute_distances(points_2d):
        # points_2d is shape (N, 2)
        diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists
    
    def objective_and_constraints(points_flat):
        points_2d = points_flat.reshape(N, 2)
        dists = compute_distances(points_2d)
        # Min distances between pairs
        iu = np.triu_indices(N, k=1)
        min_pair_dists = dists[iu]
        # Distances to boundary
        radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
        boundary_dists = 1.0 - radii
        
        # We want to maximize the minimum of all these distances / 2
        # Use a smooth approximation: soft-min via log-sum-exp
        # For maximizing min, we can minimize -min, but that's non-smooth
        # Instead, use the fact that we want all distances >= 2*r
        # So we want to maximize the minimum distance constraint
        
        # Use a penalty-based approach: minimize the sum of violations
        # Or better: use a smooth surrogate for the minimum
        
        # Let's use the log-sum-exp trick for a smooth approximation of the minimum
        # min(x) ≈ -log(exp(-sum(x))) but that's for products
        # Actually, for min of a set, we can use:
        # min(x_i) ≈ (1/beta) * log(sum(exp(-beta * x_i)))  -- this is not quite right
        
        # Better approach: use the fact that we want to maximize r such that
        # all pair distances >= 2r and all boundary distances >= r
        # So define a function that returns the "bottleneck" value
        # and use a smooth approximation
        
        # Smooth minimum approximation using log-sum-exp:
        # soft_min(x, beta) = -(1/beta) * log(sum(exp(-beta * x_i)))
        # This approximates min(x_i) as beta -> infinity
        
        beta = 50.0
        all_dists = np.concatenate([min_pair_dists, boundary_dists])
        # We want to maximize the minimum of all_dists / 2
        # So minimize -min(all_dists/2)
        # soft_min(all_dists/2, beta) ≈ min(all_dists/2)
        # So objective = -soft_min(all_dists/2, beta)
        
        scaled_dists = all_dists / 2.0
        # Numerically stable log-sum-exp
        max_val = np.max(-beta * scaled_dists)
        log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * scaled_dists - max_val)))
        soft_min = -log_sum_exp / beta
        
        return -soft_min  # We want to minimize this (i.e., maximize the soft-min)
    
    # Use gradient-based optimization with the smooth surrogate
    x0 = initial_points.flatten()
    
    # Try L-BFGS-B which can handle this
    result = minimize(
        objective_and_constraints,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    # Check time
    if time.time() > deadline:
        return initial_points
    
    optimized_points = result.x.reshape(N, 2)
    
    # Verify validity
    def is_valid(points_2d):
        if not np.all(np.isfinite(points_2d)):
            return False
        radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
        if np.any(radii > 1.0):
            return False
        dists = compute_distances(points_2d)
        iu = np.triu_indices(N, k=1)
        if np.any(dists[iu] < 1e-10):
            return False
        return True
    
    if is_valid(optimized_points):
        # Compute the actual score
        dists = compute_distances(optimized_points)
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        radii = np.hypot(optimized_points[:, 0], optimized_points[:, 1])
        min_boundary = np.min(1.0 - radii)
        score = min(min_pair, min_boundary)
        
        # Compare with initial
        dists_init = compute_distances(initial_points)
        iu = np.triu_indices(N, k=1)
        min_pair_init = np.min(dists_init[iu]) / 2.0
        radii_init = np.hypot(initial_points[:, 0], initial_points[:, 1])
        min_boundary_init = np.min(1.0 - radii_init)
        score_init = min(min_pair_init, min_boundary_init)
        
        if score > score_init:
            return optimized_points
        else:
            return initial_points
    else:
        return initial_points
    
# EVOLVE-BLOCK-END
