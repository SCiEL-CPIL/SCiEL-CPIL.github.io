"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import itertools

def construct():
    # 3D Lennard-Jones-style repulsion minimization with wall constraints
    # to maximize the minimum inter-particle distance.
    
    # 1. Initialization: Jittered Grid
    # A jittered grid provides a better starting point than pure random,
    # as it ensures a more uniform distribution of points.
    # For N=100, a 5x5x4 grid = 100 points is close. Let's use 5x5x4.
    # Or simply generate a grid of size ceil(N^(1/3)) and take first N.
    side = int(np.ceil(N ** (1.0/3.0)))
    # Create grid points
    grid_x = np.linspace(-0.4, 0.4, side)
    grid_y = np.linspace(-0.4, 0.4, side)
    grid_z = np.linspace(-0.4, 0.4, side)
    
    # Generate all combinations
    points = []
    for x in grid_x:
        for y in grid_y:
            for z in grid_z:
                points.append([x, y, z])
    
    points = np.array(points)
    
    # If we have more points than N, take the first N.
    # If we have fewer, we need to add more or adjust grid.
    # For N=100, side=5 gives 125 points. We take the first 100.
    # This creates a slightly non-uniform shape if we just slice, but it's a good start.
    # To be more uniform, we could randomly sample from the grid if grid > N.
    if len(points) > N:
        # Randomly select N points from the grid to avoid bias from slicing order
        indices = np.random.choice(len(points), N, replace=False)
        positions = points[indices]
    else:
        # If grid is too small, pad with random points (unlikely for N=100 with side=5)
        positions = np.random.uniform(-0.4, 0.4, size=(N, 3))
        
    # Add small random jitter to break symmetry
    positions += np.random.uniform(-0.01, 0.01, size=(N, 3))
    
    # Clip to ensure inside bounds with margin
    positions = np.clip(positions, -0.45, 0.45)
    
    bounds = Bounds(np.full(N * 3, -0.5), np.full(N * 3, 0.5))
    
    # Objective function: Minimize sum of 1/d^alpha
    # We will run multiple stages with increasing alpha.
    
    def energy(pos, alpha):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        iu = np.triu_indices(N, k=1)
        d_sq = dist_sq[iu]
        
        # Repulsive energy
        repulsion = np.sum(d_sq ** (-alpha / 2.0))
        
        # Wall penalty: Keep points away from walls to allow larger radius
        # Penalty if |v| > 0.5 - margin
        # We want to maximize min(0.5 - |v|), so we penalize approaching 0.5
        # Let's use a soft barrier that becomes steep near the wall.
        # E_wall = sum (max(0, |v| - (0.5 - r_est))^2) is hard because r_est changes.
        # Instead, just keep them well inside. The repulsion will push them to center anyway.
        # A simple quadratic penalty for exceeding 0.45 helps.
        abs_pos = np.abs(pos)
        wall_excess = np.maximum(abs_pos - 0.45, 0.0)
        wall_penalty = np.sum(wall_excess ** 2)
        
        return repulsion + 1000.0 * wall_penalty
    
    def gradient(pos, alpha):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        dist = np.sqrt(dist_sq)
        # d/dx_i (1/|x_i - x_j|^alpha) = -alpha * (x_i - x_j) / |x_i - x_j|^(alpha+2)
        # grad_i = sum_{j != i} -alpha * (pos_i - pos_j) / |pos_i - pos_j|^(alpha+2)
        
        # To avoid overflow/underflow issues with high alpha, compute carefully.
        # coeff = -alpha / dist^(alpha+2)
        # dist^(alpha+2) = (dist_sq)^(alpha/2 + 1)
        power = alpha / 2.0 + 1.0
        dist_pow = dist_sq ** power
        np.fill_diagonal(dist_pow, 1.0)
        
        coeff = -alpha / dist_pow
        np.fill_diagonal(coeff, 0.0)
        
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Wall gradient
        abs_pos = np.abs(pos)
        wall_excess = np.maximum(abs_pos - 0.45, 0.0)
        wall_grad = np.zeros_like(pos)
        mask = abs_pos > 0.45
        wall_grad[mask] = 2.0 * wall_excess[mask] * np.sign(pos[mask])
        
        grad += 1000.0 * wall_grad
        
        return grad

    best_pos = positions.copy()
    best_min_dist = 0.0
    
    # Multi-stage optimization
    # Stage 1: Soft repulsion (alpha=4) to arrange general structure
    # Stage 2: Medium repulsion (alpha=8) to tighten
    # Stage 3: Hard repulsion (alpha=16) to maximize min distance
    
    alphas = [4, 8, 16, 32]
    
    for alpha in alphas:
        # Run minimization
        res = minimize(
            energy,
            best_pos.ravel(),
            args=(alpha,),
            jac=lambda x, a=alpha: gradient(x, a),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        best_pos = res.x.reshape(N, 3)
        
        # Calculate current min distance
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(N, k=1)
        min_dist_sq = dist_sq[iu].min()
        min_dist = np.sqrt(min_dist_sq)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
    
    # Final Polish: Use a very high alpha to squeeze the closest pairs
    # Also, we can try to specifically push the closest pair apart.
    # For now, just run one more high-alpha step.
    
    res = minimize(
        energy,
        best_pos.ravel(),
        args=(64,),
        jac=lambda x: gradient(x, 64),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
    )
    best_pos = res.x.reshape(N, 3)
    
    # Ensure strictly inside
    best_pos = np.clip(best_pos, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_pos
# EVOLVE-BLOCK-END
