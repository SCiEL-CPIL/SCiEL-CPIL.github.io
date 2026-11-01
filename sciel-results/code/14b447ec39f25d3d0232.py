import numpy as np
from scipy.optimize import minimize
import time

N = 32

def sum_radii(c):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii of the Voronoi cells clipped to [0,1]^2
    """
    p = c.reshape(N, 2)
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p) # shape (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1) # shape (N,)
    
    # Distances to other centers / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def create_grid_init(rows, cols, n):
    """Create a grid initialization with specified rows and cols, taking first n points"""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points).flatten()

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    best_c = None
    best_val = -np.inf
    
    # Try multiple different initializations and optimization methods
    
    # Strategy 1: Regular grid (6x6 = 36, take 32)
    init_c = create_grid_init(6, 6, N)
    
    # Strategy 2: Slightly asymmetric grids
    init_variants = [
        create_grid_init(6, 6, N),
        create_grid_init(5, 7, N),
        create_grid_init(7, 5, N),
        create_grid_init(4, 8, N),
        create_grid_init(8, 4, N),
    ]
    
    # Add some random perturbations to each
    all_inits = []
    for init in init_variants:
        all_inits.append(init.copy())
        for noise_scale in [0.02, 0.05, 0.1]:
            noise = np.random.normal(0, noise_scale, size=init.shape)
            perturbed = np.clip(init + noise, 0.01, 0.99)
            all_inits.append(perturbed)
    
    # Also add pure random inits
    for _ in range(5):
        random_c = np.random.uniform(0.05, 0.95, size=N*2)
        all_inits.append(random_c)
    
    # Try each initialization with Powell and Nelder-Mead
    for idx, c_init in enumerate(all_inits):
        if time.time() - start_time > timeout:
            break
        
        # Try Powell
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, method='Powell', 
                          options={'maxiter': 300, 'disp': False})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        if time.time() - start_time > timeout:
            break
        
        # Try Nelder-Mead if we have time
        if time.time() - start_time < timeout - 30:
            try:
                res = minimize(lambda c: -sum_radii(c), c_init, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
    
    # If no valid result, fall back to grid
    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        best_val = sum_radii(best_c)
    
    # Final refinement with Powell on the best solution
    if time.time() - start_time < timeout - 10:
        try:
            res = minimize(lambda c: -sum_radii(c), best_c, method='Powell',
                          options={'maxiter': 200, 'disp': False})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
    
    # Convert to (N, 3) array
    p = best_c.reshape(N, 2)
    # Recalculate radii for the final centers
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
