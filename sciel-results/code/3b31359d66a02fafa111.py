import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit
    
    best_radius = -1
    best_centers = None
    
    # Helper to compute the radius for a given set of centers
    def compute_radius(centers):
        # centers is (N, 3)
        # Distance to walls
        # Cube is [-0.5, 0.5]^3
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        # Pairwise distances
        # Use broadcasting to compute all pairwise distances
        # dists[i, j] = ||centers[i] - centers[j]||
        dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radius = min(min_pair_dist / 2.0, min_wall_dist)
        return radius

    # Objective function to minimize (negative radius)
    def objective(x):
        centers = x.reshape(N, 3)
        # Ensure centers are within bounds? 
        # L-BFGS-B handles bounds, so we can set bounds to [-0.5, 0.5]
        r = compute_radius(centers)
        return -r

    # Define bounds for each coordinate
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Try multiple random starts
    num_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Random start inside the cube
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        x0 = centers.flatten()
        
        # Run optimization
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < -best_radius:
                r = -res.fun
                if r > best_radius:
                    best_radius = r
                    best_centers = res.x.reshape(N, 3)
        except Exception:
            pass
            
    # If no valid result was found (should not happen), return grid
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
