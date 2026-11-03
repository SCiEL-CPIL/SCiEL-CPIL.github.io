"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Objective: maximize minimum radius
    # r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    
    def compute_radius(centers):
        # centers is shape (N, 2)
        # Compute pairwise distances
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        # Square is [-0.5, 0.5]^2
        # Distance from point (x,y) to boundary is min(0.5 - |x|, 0.5 - |y|)
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r
    
    def negative_radius(centers):
        # We minimize -r
        r = compute_radius(centers)
        return -r
    
    # Initialize with grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    # Flatten for optimizer
    x0 = initial_centers.flatten()
    
    # Use Nelder-Mead for local optimization
    # This is a nonsmooth objective, but NM can handle it
    best_result = None
    best_r = compute_radius(initial_centers)
    
    # Try a few restarts with different initializations
    for restart in range(5):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            x_curr = x0.copy()
        else:
            # Jitter the initial grid
            jitter = np.random.uniform(-0.01, 0.01, size=x0.shape)
            x_curr = x0 + jitter
        
        result = minimize(
            negative_radius, 
            x_curr, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        centers = result.x.reshape(-1, 2)
        r = compute_radius(centers)
        
        if r > best_r:
            best_r = r
            best_result = centers
        
        if time.time() > deadline - 10:
            break
    
    if best_result is not None:
        return best_result
    else:
        return initial_centers
# EVOLVE-BLOCK-END
