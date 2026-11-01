import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    # Initial configuration: Jittered Grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 100 indices
    indices = np.random.choice(len(grid), N, replace=False)
    c = grid[indices].copy()
    
    # Jitter
    c += np.random.normal(0, 0.02, c.shape)
    c = np.clip(c, -0.5, 0.5)
    
    best_c = c.copy()
    best_r = _compute_r(best_c)
    
    # Optimization Parameters
    lr = 0.005
    lambda_wall = 100.0
    stagnation_limit = 2000
    stagnation_counter = 0
    restart_jitter = 0.05
    
    while time.time() - start_time < time_limit:
        # Compute distances and forces
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        dists_sq = np.maximum(dists**2, 1e-10)
        inv_dists_sq = 1.0 / dists_sq
        np.fill_diagonal(inv_dists_sq, 0)
        
        # Pairwise repulsive force
        term = 2.0 * inv_dists_sq[:, :, np.newaxis] * diffs
        net_force = np.sum(term, axis=1)
        
        # Wall force
        abs_c = np.abs(c)
        over = np.clip(abs_c - 0.5, 0.0, None)
        wall_force = lambda_wall * 2.0 * np.sign(c) * over
        
        total_force = net_force + wall_force
        
        # Update
        new_c = c - lr * total_force
        new_c = np.clip(new_c, -0.5, 0.5)
        
        new_r = _compute_r(new_c)
        
        if new_r > best_r:
            best_c = new_c.copy()
            best_r = new_r
            c = new_c
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            # Reduce learning rate
            lr *= 0.95
            
            if stagnation_counter > stagnation_limit:
                # Restart from best with jitter
                c = best_c + np.random.normal(0, restart_jitter, best_c.shape)
                c = np.clip(c, -0.5, 0.5)
                lr = 0.005
                stagnation_counter = 0
                restart_jitter *= 0.9  # Decay jitter for future restarts
                if restart_jitter < 0.001:
                    restart_jitter = 0.001

    return best_c

def _compute_r(c):
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    abs_c = np.abs(c)
    clearances = 0.5 - abs_c.max(axis=1)
    min_wall_clearance = np.min(clearances)
    
    return min(min_pair_dist / 2.0, min_wall_clearance)
