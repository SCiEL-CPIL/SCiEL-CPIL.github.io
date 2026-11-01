"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the max common radius for given centers in [-0.5, 0.5]^3."""
    # Distance to walls
    wall_dists = np.min(np.abs(centers), axis=1)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_min)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return min_wall * 2
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_radius = 0.0
    best_centers = None
    
    # Initial: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    r = _compute_radius(centers)
    if r > best_radius:
        best_radius = r
        best_centers = centers.copy()
    
    # Try different initial configurations and optimize
    rng = np.random.RandomState(42)
    
    # Strategy 1: Random perturbation + optimization
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Start from a slightly perturbed grid
        c = centers.copy()
        c += rng.normal(0, 0.01, c.shape)
        # Clip to stay inside cube
        c = np.clip(c, -0.49, 0.49)
        
        # Use L-BFGS-B to maximize the minimum distance (equivalently, minimize negative)
        def objective(x):
            if time.time() > deadline:
                return 1e10
            c = x.reshape(N, 3)
            # Soft penalty for going outside
            outside = np.maximum(0, np.abs(c) - 0.5)
            if np.any(outside > 0.001):
                return 1e10 + np.sum(outside ** 2)
            return -_compute_radius(c)
        
        try:
            res = minimize(objective, c.ravel(), method='L-BFGS-B', 
                          options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-8})
            if res.success or res.fun < 1e9:
                c_opt = res.x.reshape(N, 3)
                if np.all(np.abs(c_opt) <= 0.5 + 1e-10):
                    c_opt = np.clip(c_opt, -0.5, 0.5)
                    r_opt = _compute_radius(c_opt)
                    if r_opt > best_radius:
                        best_radius = r_opt
                        best_centers = c_opt.copy()
        except Exception:
            pass
    
    # Strategy 2: Start from a better structured initial guess
    # Use a body-centered cubic inspired arrangement
    if time.time() < deadline:
        # Create a denser grid and select best subset
        m2 = int(np.ceil((N * 1.5) ** (1.0 / 3.0) - 1e-9))
        ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
        grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
        
        if len(grid2) >= N:
            # Randomly sample N points from the denser grid
            idx = rng.choice(len(grid2), N, replace=False)
            c2 = grid2[idx].copy()
            
            def objective2(x):
                if time.time() > deadline:
                    return 1e10
                c = x.reshape(N, 3)
                outside = np.maximum(0, np.abs(c) - 0.5)
                if np.any(outside > 0.001):
                    return 1e10 + np.sum(outside ** 2)
                return -_compute_radius(c)
            
            try:
                res2 = minimize(objective2, c2.ravel(), method='L-BFGS-B',
                               options={'maxiter': 300, 'maxfun': 8000, 'gtol': 1e-8})
                if res2.success or res2.fun < 1e9:
                    c_opt2 = res2.x.reshape(N, 3)
                    if np.all(np.abs(c_opt2) <= 0.5 + 1e-10):
                        c_opt2 = np.clip(c_opt2, -0.5, 0.5)
                        r_opt2 = _compute_radius(c_opt2)
                        if r_opt2 > best_radius:
                            best_radius = r_opt2
                            best_centers = c_opt2.copy()
            except Exception:
                pass
    
    # Final polish: one more optimization pass on the best found
    if best_centers is not None and time.time() < deadline:
        def objective_final(x):
            if time.time() > deadline:
                return 1e10
            c = x.reshape(N, 3)
            outside = np.maximum(0, np.abs(c) - 0.5)
            if np.any(outside > 0.001):
                return 1e10 + np.sum(outside ** 2)
            return -_compute_radius(c)
        
        try:
            res_final = minimize(objective_final, best_centers.ravel(), method='L-BFGS-B',
                                options={'maxiter': 500, 'maxfun': 10000, 'gtol': 1e-9})
            if res_final.success or res_final.fun < 1e9:
                c_final = res_final.x.reshape(N, 3)
                if np.all(np.abs(c_final) <= 0.5 + 1e-10):
                    c_final = np.clip(c_final, -0.5, 0.5)
                    r_final = _compute_radius(c_final)
                    if r_final > best_radius:
                        best_radius = r_final
                        best_centers = c_final.copy()
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback to original grid
        return grid[:N]
    
    # Ensure validity: clip to cube and verify
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
