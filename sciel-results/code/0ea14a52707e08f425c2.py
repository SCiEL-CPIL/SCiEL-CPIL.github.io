import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
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
    time_limit = 145.0
    
    def get_radius(c):
        return _compute_radius(c)

    # Initialization: 5x5x4 Grid, inset slightly
    def initial_grid():
        nx, ny, nz = 5, 5, 4
        # Inset by 0.05 to avoid starting on the boundary with r=0
        # This helps SA start with a positive radius
        inset = 0.05
        xs = np.linspace(-0.5 + inset, 0.5 - inset, nx)
        ys = np.linspace(-0.5 + inset, 0.5 - inset, ny)
        zs = np.linspace(-0.5 + inset, 0.5 - inset, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return grid

    best_c = initial_grid()
    best_r = get_radius(best_c)
    
    # Simulated Annealing
    T = 0.005
    T_min = 0.00005
    alpha = 0.9995
    step_size = 0.05
    
    c_current = best_c.copy()
    r_current = best_r
    
    # Track stagnation
    last_improvement_time = start_time
    stagnation_count = 0
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        noise = np.random.normal(0, step_size, size=(N, 3))
        c_new = c_current + noise
        
        r_new = get_radius(c_new)
        
        # Acceptance criterion
        if r_new > r_current or (T > 0 and np.random.random() < np.exp((r_new - r_current) / T)):
            c_current = c_new
            r_current = r_new
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                last_improvement_time = time.time()
                stagnation_count = 0
            else:
                stagnation_count += 1
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # If stuck, restart with higher T
            if stagnation_count > 1000:
                T = 0.005
                step_size = 0.05
                stagnation_count = 0

    # Final Polish with L-BFGS-B on the best solution found
    if time.time() - start_time < time_limit - 5:
        def smooth_objective(x, beta=100.0):
            c = x.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            dists = np.sqrt(dists_sq)
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
        res = minimize(
            lambda x: smooth_objective(x, beta=100.0),
            best_c.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfev': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        c_test = res.x.reshape(N, 3)
        r_test = get_radius(c_test)
        if r_test > best_r:
            best_r = r_test
            best_c = c_test
    
    return best_c
