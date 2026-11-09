import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_centers = None
    best_radius = 0.0

    def calc_exact_radius(centers):
        """Calculate the exact minimum radius for a given configuration."""
        # Pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # Distance to each wall: 0.5 - |coord|
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def soft_min_objective(flat_x, beta):
        """
        Objective function for L-BFGS-B.
        Minimizes the negative soft-min of all constraint distances.
        """
        centers = flat_x.reshape(N, 3)
        
        # 1. Pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        
        # 2. Wall distances
        wall_dists = 0.5 - np.abs(centers)
        wall_vals = wall_dists.flatten()
        
        # Combine all constraints
        # To save memory and time, we can compute the soft min in parts
        # or just concatenate. N=100 -> 4950 pairs + 300 walls = 5250 values.
        # This is small enough to concatenate.
        all_vals = np.concatenate([pair_vals[~np.eye(N, dtype=bool)], wall_vals])
        
        # Soft min calculation with log-sum-exp trick
        min_val = np.min(all_vals)
        # If min_val is very small (close to 0), exp(-beta * (val - min_val)) is stable
        # exp(-beta * 0) = 1
        exp_terms = np.exp(-beta * (all_vals - min_val))
        sum_exp = np.sum(exp_terms)
        
        soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
        
        # We want to maximize soft_min_val, so we minimize its negative
        return -soft_min_val

    # Bounding box for L-BFGS-B
    bounds = [(-0.5, 0.5)] * (N * 3)

    # Beta schedule: start smooth, get sharper
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    # Number of restarts
    num_restarts = 20
    
    # Initial grid generation helper
    def get_initial_centers(noise_scale):
        # 5x5x5 grid = 125 points. We need 100.
        # We take the first 100 after shuffling indices to break symmetry randomly
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        # Randomly select 100 points from the 125 grid points
        # This ensures we don't always bias towards a corner
        indices = np.random.choice(125, 100, replace=False)
        centers = grid[indices].copy()
        
        # Add noise
        centers += np.random.normal(0, noise_scale, centers.shape)
        
        # Clip to bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        return centers

    for r_idx in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Vary noise scale for diversity
        noise_scale = 0.01 + 0.02 * (r_idx % 3)
        
        x0 = get_initial_centers(noise_scale)
        
        # Annealing loop
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Define objective for current beta
            def objective(flat_x):
                return soft_min_objective(flat_x, beta)
            
            # Run L-BFGS-B
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6})
                
                if res.success or res.fun < 0: # res.fun is -soft_min, so lower is better
                    new_centers = res.x.reshape(N, 3)
                    new_radius = calc_exact_radius(new_centers)
                    
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centers = new_centers.copy()
                    
                    # Use the optimized point as start for next beta
                    x0 = new_centers
                    
            except Exception:
                # If optimization fails, keep current x0
                pass
                
            # Check time after each beta step
            if time.time() > deadline:
                break

    # If no optimization succeeded, return the best grid we had? 
    # Or just the initial one?
    if best_centers is None:
        # Fallback to a simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        
    return best_centers
