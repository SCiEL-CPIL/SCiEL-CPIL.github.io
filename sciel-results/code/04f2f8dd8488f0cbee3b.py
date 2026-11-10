import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
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
    
    def soft_min_objective(x, beta):
        """
        Smooth objective: -soft_min(min_pair_dist/2, min_wall_dist)
        Using log-sum-exp trick for numerical stability.
        """
        c = x.reshape(N, 3)
        # Pairwise distances / 2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        
        # Wall distances
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        wall_vals = 0.5 - max_coord
        
        # Combine all constraints
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

    def force_directed_refine(c, time_budget):
        """
        Force-directed refinement: push spheres apart and away from walls.
        Uses actual (non-smooth) radius for acceptance.
        """
        c = np.array(c, dtype=np.float64)
        n = len(c)
        best_c = c.copy()
        best_r = _compute_radius(c)
        
        end_time = time.time() + time_budget
        step_size = 0.005
        max_iter = 2000
        
        for it in range(max_iter):
            if time.time() > end_time:
                break
            
            # Compute forces
            forces = np.zeros_like(c)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    d = c[i] - c[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-10:
                        forces[i] += np.random.randn(3) * 0.001
                        forces[j] -= np.random.randn(3) * 0.001
                    else:
                        f = d / (dist * dist)
                        forces[i] += f
                        forces[j] -= f
            
            # Wall attraction (push away from walls)
            for i in range(n):
                for k in range(3):
                    dist_to_wall = 0.5 - np.abs(c[i, k])
                    if dist_to_wall < 0.05:
                        forces[i, k] += np.sign(c[i, k]) * (0.05 - dist_to_wall) * 10.0
            
            # Normalize forces
            norm_f = np.linalg.norm(forces, axis=1)
            for i in range(n):
                if norm_f[i] > 1.0:
                    forces[i] /= norm_f[i]
            
            # Move
            new_c = c + step_size * forces
            new_c = np.clip(new_c, -0.5 + 1e-10, 0.5 - 1e-10)
            
            # Check if improved
            new_r = _compute_radius(new_c)
            
            if new_r > best_r:
                c = new_c
                best_c = c.copy()
                best_r = new_r
                step_size = min(step_size * 1.05, 0.02)
            else:
                step_size *= 0.95
                if step_size < 1e-6:
                    break
        
        return best_c, best_r

    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Start with a 5x5x5 grid, take first 100, add jitter
        ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        jitter = rng.uniform(-0.02, 0.02, size=(N, 3))
        grid = grid + jitter
        grid = np.clip(grid, -0.5, 0.5)
        return grid.flatten()

    best_r = 0.0
    best_c = None
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Strategy:
    # 1. Multiple restarts with different initializations
    # 2. For each: force-directed refinement, then smooth gradient optimization
    # 3. Track best result
    
    n_restarts = 8
    time_per_restart = (time_limit - 10.0) / n_restarts
    
    for seed in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        x0 = make_initial_guess(seed)
        
        # Phase 1: Force-directed refinement (global exploration)
        time_fd = min(time_per_restart * 0.4, 10.0)
        c_fd, r_fd = force_directed_refine(x0.reshape(N, 3), time_fd)
        
        if r_fd > best_r:
            best_r = r_fd
            best_c = c_fd.copy()
        
        # Phase 2: Smooth gradient optimization (local refinement)
        if time.time() - start_time < time_limit - 5:
            # Use a moderate beta for smooth optimization
            beta = 50.0
            res = minimize(
                lambda x: soft_min_objective(x, beta),
                c_fd.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfev': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            c_test = res.x.reshape(N, 3)
            r_test = _compute_radius(c_test)
            
            if r_test > best_r:
                best_r = r_test
                best_c = c_test.copy()
        
        # Phase 3: High-precision polish with tighter beta
        if time.time() - start_time < time_limit - 5:
            beta_high = 200.0
            res2 = minimize(
                lambda x: soft_min_objective(x, beta_high),
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfev': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            c_test2 = res2.x.reshape(N, 3)
            r_test2 = _compute_radius(c_test2)
            
            if r_test2 > best_r:
                best_r = r_test2
                best_c = c_test2.copy()
    
    # Final fallback: if we haven't improved much, try a few more random starts
    if time.time() - start_time < time_limit - 10:
        for seed in range(n_restarts, n_restarts + 4):
            if time.time() - start_time > time_limit - 5:
                break
            
            x0 = make_initial_guess(seed)
            c_fd, r_fd = force_directed_refine(x0.reshape(N, 3), 5.0)
            
            if r_fd > best_r:
                best_r = r_fd
                best_c = c_fd.copy()
            
            if time.time() - start_time < time_limit - 5:
                beta = 100.0
                res = minimize(
                    lambda x: soft_min_objective(x, beta),
                    c_fd.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'maxfev': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                c_test = res.x.reshape(N, 3)
                r_test = _compute_radius(c_test)
                
                if r_test > best_r:
                    best_r = r_test
                    best_c = c_test.copy()
    
    if best_c is None:
        # Fallback: simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
