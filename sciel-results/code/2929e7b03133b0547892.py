"""Program for packing 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Find centers of 50 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the minimum distance to walls and between circles.
    
    Score = min(
        min_{i<j} ||c_i - c_j|| / 2,
        min_i min_k (0.5 - |c_{ik}|)
    )
    """
    # Initial guess: grid layout
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].flatten()
    
    # Objective: maximize the minimum constraint value
    # We'll minimize the negative of the minimum radius
    # r = min(
    #     min_{i<j} ||c_i - c_j|| / 2,
    #     min_i min_k (0.5 - |c_{ik}|)
    # )
    
    def objective(x):
        """Return negative of the minimum radius (to minimize)."""
        centers = x.reshape(N, 2)
        
        # Wall constraints: 0.5 - |c_{ik}|
        # For each center, the distance to nearest wall
        wall_dists = np.min(np.stack([
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1])
        ]), axis=1)
        
        # Pairwise distances / 2
        # Compute all pairwise distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Minimum of all constraints
        r = min(np.min(wall_dists), min_pair_dist)
        
        return -r
    
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use SLSQP to maximize the minimum radius
    # We minimize -r
    
    best_x = x0.copy()
    best_score = objective(best_x)
    
    # Try multiple random initializations to avoid local optima
    np.random.seed(42)
    
    for trial in range(20):
        # Perturb the initial guess
        if trial == 0:
            x_start = x0.copy()
        else:
            # Random perturbation around grid
            x_start = x0 + np.random.randn(2 * N) * 0.02
            # Clip to bounds
            x_start = np.clip(x_start, -0.5, 0.5)
        
        # Also try a few fully random starts
        if trial > 10:
            x_start = np.random.uniform(-0.5, 0.5, size=2 * N)
        
        try:
            result = minimize(
                objective,
                x_start,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if result.success or (not np.isnan(result.fun) and result.fun < best_score):
                if result.fun < best_score:
                    best_score = result.fun
                    best_x = result.x
        except Exception:
            continue
    
    # Final refinement with tighter tolerance
    try:
        result = minimize(
            objective,
            best_x,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if not np.isnan(result.fun) and result.fun < best_score:
            best_score = result.fun
            best_x = result.x
    except Exception:
        pass
    
    centers = best_x.reshape(N, 2)
    
    # Ensure centers are within bounds (clip slightly inside)
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers

# EVOLVE-BLOCK-END
