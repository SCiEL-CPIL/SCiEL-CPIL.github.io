import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s margin

def compute_radius(centers):
    """
    Compute the maximum radius for the given centers.
    centers: (N, 3) array
    Returns: radius (float)
    """
    # Distances between all pairs
    # To avoid memory explosion for larger N, we can do it in chunks or just vectorize for N=50 (50*50/2 = 1225 pairs, trivial)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distances to walls
    # Centers must be in [-0.5, 0.5]
    # dist to wall for coord k is 0.5 - |c_k|
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half the min distance between spheres, and min distance to wall
    # Wait, the problem says:
    # r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    # So if min_pair_dist is d, radius is d/2.
    # If min_wall_dist is w, radius is w.
    # The spheres are disjoint if 2*r <= min_pair_dist -> r <= min_pair_dist/2.
    # Spheres inside if r <= min_wall_dist.
    # So r = min(min_pair_dist / 2.0, min_wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def objective(x_flat):
    centers = x_flat.reshape(N, 3)
    r = compute_radius(centers)
    # We want to maximize r, so minimize -r
    # If r is negative (center outside), -r is positive, which is bad for minimization.
    # This correctly penalizes going outside.
    return -r

def construct():
    global START_TIME, TIME_LIMIT
    
    best_centers = None
    best_radius = -np.inf
    
    # Strategy: Multiple restarts with Nelder-Mead
    
    # 1. Grid initialization (perturbed)
    # 2. Random initializations
    
    num_restarts = 15  # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        if i == 0:
            # Grid start
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            centers = grid[:N].copy()
            # Add small random noise to break symmetry and help Nelder-Mead
            centers += np.random.normal(0, 0.01, centers.shape)
            # Clamp to ensure valid start
            centers = np.clip(centers, -0.49, 0.49)
        else:
            # Random start
            # Start slightly inside to avoid immediate invalid states
            centers = np.random.uniform(-0.45, 0.45, (N, 3))
        
        x0 = centers.flatten()
        
        # Nelder-Mead options
        # maxiter: limit iterations per restart to prevent hanging
        # xatol, fatol: tolerance for convergence
        options = {
            'maxiter': 2000,
            'xatol': 1e-6,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=options)
            
            if res.success or res.fun < -best_radius: # res.fun is -r, so lower is better
                r_found = -res.fun
                if r_found > best_radius:
                    best_radius = r_found
                    best_centers = res.x.reshape(N, 3)
                    
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback to seed if all failed
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
