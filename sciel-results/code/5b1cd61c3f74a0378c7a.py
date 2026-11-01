import numpy as np
import time
import scipy.optimize as opt

N = 50
CUBE_HALF = 0.5

def compute_radius(centers):
    # centers: (N, 3)
    # Compute min distance between centers
    # and min distance to boundary
    # dist_to_boundary_i = 0.5 - max(abs(c_ix), abs(c_iy), abs(c_iz))
    
    # Vectorized distance to boundary
    max_abs = np.max(np.abs(centers), axis=1)
    dist_to_boundary = CUBE_HALF - max_abs
    
    # Pairwise distances
    # To save memory/time for N=50, we can compute full distance matrix
    # or just iterate. 50*49/2 = 1225 pairs. Full matrix is 50x50.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = 0.5 * np.min([min_pair_dist, np.min(dist_to_boundary)])
    return r

def energy(centers_flat, p=2.0):
    centers = centers_flat.reshape(-1, 3)
    # Distance to boundary
    max_abs = np.max(np.abs(centers), axis=1)
    dist_to_boundary = CUBE_HALF - max_abs
    # Avoid division by zero
    dist_to_boundary = np.maximum(dist_to_boundary, 1e-10)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, 1e-10) # Avoid div by 0
    
    # We only want upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Sum of inverse powers
    e_pair = np.sum(1.0 / (pair_dists ** p))
    e_bound = np.sum(1.0 / (dist_to_boundary ** p))
    
    return e_pair + e_bound

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_centers = None
    best_radius = -1.0
    
    # Initial guess: Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    # Random restarts
    num_restarts = 50 # We will break early if time runs out
    np.random.seed(42) # Reproducibility
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb initial centers
        if restart == 0:
            centers = initial_centers.copy()
        else:
            # Random perturbation
            # Scale perturbation based on expected radius (~0.13)
            # Don't make it too large to keep it in the basin of the grid-like solution
            # Or use full random? Grid is a good start.
            centers = initial_centers + np.random.randn(N, 3) * 0.05
            # Clamp to cube
            centers = np.clip(centers, -CUBE_HALF, CUBE_HALF)
        
        # Optimization
        # Use L-BFGS-B
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        
        # We maximize radius, so minimize negative energy?
        # Actually, minimizing the energy function (sum of 1/d^p) correlates with maximizing min distance.
        
        try:
            res = opt.minimize(
                energy, 
                centers.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_centers = res.x.reshape(-1, 3)
            r_opt = compute_radius(opt_centers)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = opt_centers.copy()
                
        except Exception:
            pass
            
    # Final Polish
    # If we have a good candidate, try to polish it with a tighter tolerance
    # and maybe a different method or just more iterations
    if best_centers is not None:
        # Run again with tighter tol
        try:
            res = opt.minimize(
                energy, 
                best_centers.flatten(), 
                method='L-BFGS-B', 
                bounds=[(-CUBE_HALF, CUBE_HALF)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            opt_centers = res.x.reshape(-1, 3)
            r_opt = compute_radius(opt_centers)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = opt_centers.copy()
        except Exception:
            pass
            
    # If no valid center found (should not happen), return grid
    if best_centers is None:
        return initial_centers
        
    return best_centers

# EVOLVE-BLOCK-END
