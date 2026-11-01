import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Initial grid: 5x5x4
    # 5*5*4 = 100
    m_x, m_y, m_z = 5, 5, 4
    ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
    ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
    ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    base_grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    
    def compute_radius(centers):
        # centers: (N, 3)
        # Compute pairwise distances
        # Using broadcasting for simplicity, might be slow for large N but N=100 is fine (100*100 = 10k pairs)
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1))
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundaries
        # centers in [-0.5, 0.5]
        # dist to boundary = 0.5 - |coord|
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary_dist = np.min(boundary_dists)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def soft_min_objective(flat_centers, beta):
        centers = flat_centers.reshape(-1, 3)
        # Pairwise distances
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # We want the min of (dists/2) and (boundary_dists)
        boundary_dists = 0.5 - np.abs(centers)
        
        # Flatten the list of relevant distances
        # Only upper triangle for pairs
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        bound_dists = boundary_dists.flatten()
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * dists)))
        # To avoid overflow/underflow, subtract max
        max_d = np.max(all_dists)
        exp_term = np.exp(-beta * (all_dists - max_d))
        log_sum = np.log(np.sum(exp_term))
        soft_min_val = max_d - log_sum / beta
        
        return -soft_min_val  # We minimize negative soft min

    best_radius = -1
    best_centers = base_grid.copy()
    
    # Define bounds for L-BFGS-B
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Strategy: Multiple restarts with increasing beta
    # Start with low beta (smoother), then refine with high beta
    
    beta_schedule = [10.0, 50.0, 200.0]
    
    # Number of restarts
    num_restarts = 5
    max_time_per_restart = (time_limit - (time.time() - start_time)) / (num_restarts * len(beta_schedule))
    max_time_per_restart = min(max_time_per_restart, 10.0) # Cap per run
    
    current_centers = base_grid.copy()
    
    for restart in range(num_restarts):
        # Jitter the initial grid
        jitter_scale = 0.05 + 0.05 * restart
        current_centers = base_grid + np.random.uniform(-jitter_scale, jitter_scale, size=base_grid.shape)
        # Clip to bounds
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - 5:
                break
            
            initial_flat = current_centers.flatten()
            
            def obj_func(x):
                return soft_min_objective(x, beta)
            
            try:
                res = minimize(obj_func, initial_flat, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 100, 'maxfun': 2000, 'disp': False})
            except Exception:
                continue
            
            if res.success or res.fun < 1e10:
                new_centers = res.x.reshape(-1, 3)
                # Validate
                if np.all(np.isfinite(new_centers)):
                    r = compute_radius(new_centers)
                    if r > best_radius:
                        best_radius = r
                        best_centers = new_centers.copy()
                    current_centers = new_centers.copy()
        
        # Update best radius from current_centers if not already updated in inner loop
        r = compute_radius(current_centers)
        if r > best_radius:
            best_radius = r
            best_centers = current_centers.copy()

    # Final check and return
    final_r = compute_radius(best_centers)
    if final_r > best_radius:
        best_radius = final_r

    return best_centers
