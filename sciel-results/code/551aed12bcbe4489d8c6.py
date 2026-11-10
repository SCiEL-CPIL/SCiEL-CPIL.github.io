import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    # Wall clearance: min over i,k of (0.5 - |c_ik|)
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
    # Inter-sphere clearance: min over i<j of |c_i - c_j| / 2
    if len(centers) > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_wall
    
    return min(min_pair, min_wall)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Objective function: Sum of squared violations
    def energy(x, R):
        c = x.reshape(N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise violations
        pair_viol = np.maximum(0, 2 * R - dists)
        
        # Wall violations
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        wall_viol = np.maximum(0, max_coord - (0.5 - R))
        
        return np.sum(pair_viol**2) + np.sum(wall_viol**2)

    # Smooth objective using log-sum-exp for soft-min
    def smooth_objective(x, beta=50.0):
        c = x.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        wall_vals = 0.5 - max_coord
        
        all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
        all_vals = all_vals[np.isfinite(all_vals)]
        
        if len(all_vals) == 0:
            return 0.0
            
        max_val = np.max(all_vals)
        shifted = -beta * (all_vals - max_val)
        shifted = np.clip(shifted, -700, 0)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min_val = max_val - log_sum / beta
        
        return -soft_min_val

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # 5x5x5 grid
        ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        # Add small jitter to break symmetry
        jitter = rng.uniform(-0.01, 0.01, size=(N, 3))
        grid = grid + jitter
        grid = np.clip(grid, -0.5, 0.5)
        return grid.flatten()

    best_r = 0.0
    best_c = None
    
    # Try multiple seeds
    for seed in range(4):
        if time.time() - start_time > time_limit - 15:
            break
            
        x0 = make_initial_guess(seed)
        x_current = x0.copy()
        
        # Phase 1: Iterative radius increase with penalty method
        R = 0.08
        max_R = 0.12
        
        while R < max_R:
            if time.time() - start_time > time_limit - 10:
                break
                
            # Optimize for current R
            res = minimize(
                lambda x: energy(x, R), 
                x_current, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'maxfev': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e-8:
                # Check if it's actually valid
                c_test = res.x.reshape(N, 3)
                r_test = _compute_radius(c_test)
                
                if r_test >= R - 1e-6:
                    # Accept this configuration and radius
                    x_current = res.x
                    R += 0.002
                else:
                    # Stuck, break
                    break
            else:
                # Optimization failed or energy too high
                break
        
        # Phase 2: Polish with smooth objective
        if time.time() - start_time < time_limit - 5:
            # Increase beta for more precise min approximation
            res2 = minimize(
                lambda x: smooth_objective(x, beta=100.0),
                x_current, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'maxfev': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res2.success or res2.fun < (res.fun if 'res' in locals() else 0):
                c_test2 = res2.x.reshape(N, 3)
                r_test2 = _compute_radius(c_test2)
                
                if r_test2 > best_r:
                    best_r = r_test2
                    best_c = c_test2.copy()
        
        # Also record the result from penalty phase
        c_penalty = x_current.reshape(N, 3)
        r_penalty = _compute_radius(c_penalty)
        
        if r_penalty > best_r:
            best_r = r_penalty
            best_c = c_penalty.copy()

    if best_c is None:
        # Fallback to simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c
