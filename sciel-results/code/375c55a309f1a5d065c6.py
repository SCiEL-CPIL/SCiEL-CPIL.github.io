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
    time_limit = 110.0 # Safety margin to ensure we return before 120s
    
    def get_radius(c):
        return _compute_radius(c)

    def initial_grid():
        # 5x5x4 grid
        nx, ny, nz = 5, 5, 4
        xs = np.linspace(-0.5, 0.5, nx)
        ys = np.linspace(-0.5, 0.5, ny)
        zs = np.linspace(-0.5, 0.5, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return grid

    best_c = initial_grid()
    best_r = get_radius(best_c)
    
    # Simulated Annealing
    T = 0.01
    T_min = 0.0001
    alpha = 0.999 # Cooling rate
    step_size = 0.05
    
    c_current = best_c.copy()
    r_current = best_r
    
    # Pre-allocate for speed? No, just vectorize.
    
    while time.time() - start_time < time_limit - 10.0: # Leave 10s for polish
        # Generate perturbation
        # Move a random subset of points? Or all?
        # Moving all points with small noise is like a global translation + deformation.
        # Moving random points is better for local search.
        
        # Let's move all points with small Gaussian noise
        noise = np.random.normal(0, step_size, size=(N, 3))
        c_new = c_current + noise
        
        # Keep inside bounds? 
        # If a point goes outside, the radius becomes negative/invalid.
        # The objective function handles this by returning a small/negative value 
        # (actually _compute_radius returns min(dist/2, wall). If wall < 0, it's negative.
        # We want to maximize. So invalid states have low score.
        
        r_new = get_radius(c_new)
        
        # Acceptance criterion
        if r_new > r_current or np.random.random() < np.exp((r_new - r_current) / T):
            c_current = c_new
            r_current = r_new
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart with higher T?
            # If we are stuck, increase step size or T
            # Simple restart:
            if best_r < 0.111: # If we haven't reached near optimal
                 T = 0.01
                 step_size = 0.05

    # Final Polish with L-BFGS-B on the best solution found
    # Only run if we have enough time
    if time.time() - start_time < time_limit - 2.0:
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
        
        # Reduce iterations drastically to ensure it finishes in time
        # 50 iterations of L-BFGS-B with 300 vars is already heavy
        # Let's try 20 iterations max
        res = minimize(
            lambda x: smooth_objective(x, beta=100.0),
            best_c.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20, 'maxfev': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        c_test = res.x.reshape(N, 3)
        r_test = get_radius(c_test)
        if r_test > best_r:
            best_r = r_test
            best_c = c_test
    
    return best_c
