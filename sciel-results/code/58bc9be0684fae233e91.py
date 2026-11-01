"""Circle packing in unit square n=32."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi

N = 32
DEADLINE = 135.0  # safety margin before 150s limit


def _radii_from_centers(centers):
    """
    Calculate the maximum radius for each circle given fixed centers.
    centers: (n, 2) array.
    Returns: (n,) array of radii.
    r_i = min( distance to square boundary, min_{j != i} distance(c_i, c_j) / 2 )
    """
    n = len(centers)
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Distance to other circles
    # Compute pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dist, np.inf)
    
    # The maximum radius is half the distance to the nearest neighbor
    r_neighbor = 0.5 * np.min(dist, axis=1)
    
    # The actual radius is the minimum of boundary distance and neighbor distance
    r = np.minimum(r_bound, r_neighbor)
    
    # Ensure non-negative
    r = np.maximum(r, 0.0)
    
    return r


def _objective_centers(centers):
    """
    Objective to minimize: -sum(radii).
    centers: flattened array of shape (2*n,)
    """
    n = len(centers) // 2
    c = centers.reshape(n, 2)
    # Clip centers to [0, 1] to keep them valid, though optimizer should stay within bounds
    c = np.clip(c, 0.0, 1.0)
    r = _radii_from_centers(c)
    return -np.sum(r)


def construct():
    start_time = time.time()
    
    # We will optimize centers (2*N variables).
    # Bounds: [0, 1] for each coordinate.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_centers = None
    best_score = -np.inf
    
    def _update_best(centers):
        nonlocal best_centers, best_score
        if centers is None:
            return
        centers = np.clip(centers, 0.0, 1.0)
        r = _radii_from_centers(centers)
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    # Strategy 1: Grid initialization + Local Optimization
    # A 6x6 grid has 36 points, we can use 32 of them or just start with a 6x6 grid and let optimization move them.
    # Better: Start with a 6x6 grid of 36 points, randomly select 32? No, that creates uneven spacing.
    # Let's try a 4x8, 8x4, 5x7, 7x5, 6x6 grid.
    
    grid_configs = [
        (6, 6), # 36 points, we will use first 32 or optimize all 36? 
        # The task requires exactly N=32 circles. 
        # If we start with 36, we must drop 4. 
        # Easier: Start with a grid of exactly 32 points if possible? 4x8=32.
        (4, 8),
        (8, 4),
        (5, 7), # 35 points, drop 3
        (7, 5),
        (5, 5), # 25 points, add 7 random? 
    ]
    
    # Let's stick to configurations that are close to 32 or exactly 32.
    # (4,8) is exactly 32.
    # (6,6) is 36. We can just initialize 32 points from a 6x6 grid by taking the first 32 in row-major order? 
    # Or better, generate a 6x6 grid, and if we have more than N, randomly subsample.
    # Actually, for (4,8) it's perfect.
    
    inits = []
    
    # 1. Exact 4x8 grid
    xs = np.linspace(1/8, 7/8, 4)
    ys = np.linspace(1/16, 15/16, 8)
    # Wait, 4 cols, 8 rows.
    # x in {1/8, 3/8, 5/8, 7/8}
    # y in {1/16, 3/16, ...}
    grid_4_8 = []
    for i in range(4):
        for j in range(8):
            grid_4_8.append([(i + 0.5) / 4.0, (j + 0.5) / 8.0])
    inits.append(np.array(grid_4_8))
    
    # 2. Exact 8x4 grid
    grid_8_4 = []
    for i in range(8):
        for j in range(4):
            grid_8_4.append([(i + 0.5) / 8.0, (j + 0.5) / 4.0])
    inits.append(np.array(grid_8_4))
    
    # 3. 5x7 grid (35 points) -> take first 32
    grid_5_7 = []
    for i in range(5):
        for j in range(7):
            grid_5_7.append([(i + 0.5) / 5.0, (j + 0.5) / 7.0])
    inits.append(np.array(grid_5_7[:32]))
    
    # 4. 7x5 grid (35 points) -> take first 32
    grid_7_5 = []
    for i in range(7):
        for j in range(5):
            grid_7_5.append([(i + 0.5) / 7.0, (j + 0.5) / 5.0])
    inits.append(np.array(grid_7_5[:32]))
    
    # 5. Random inits
    for _ in range(5):
        inits.append(np.random.uniform(0.05, 0.95, size=(N, 2)))
        
    # 6. 6x6 grid (36 points) -> take first 32
    grid_6_6 = []
    for i in range(6):
        for j in range(6):
            grid_6_6.append([(i + 0.5) / 6.0, (j + 0.5) / 6.0])
    inits.append(np.array(grid_6_6[:32]))
    
    # Optimization Loop
    for init in inits:
        if time.time() - start_time > DEADLINE * 0.8:
            break
            
        init_flat = init.flatten()
        
        # Try Nelder-Mead
        try:
            res = minimize(
                _objective_centers, 
                init_flat, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            _update_best(res.x)
        except Exception:
            pass
        
        # Try Powell
        if time.time() - start_time > DEADLINE * 0.8:
            continue
        try:
            res = minimize(
                _objective_centers, 
                init_flat, 
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            _update_best(res.x)
        except Exception:
            pass

    # If we have time, do a global search with Differential Evolution on centers
    if time.time() - start_time < DEADLINE * 0.5 and best_centers is not None:
        # Use best_centers as a hint or just run DE
        # DE is slow for 64 dims, but let's try a short run
        try:
            de_res = differential_evolution(
                _objective_centers, 
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=True
            )
            _update_best(de_res.x)
        except Exception:
            pass

    # Final Polish on the best result found
    if best_centers is not None:
        if time.time() - start_time < DEADLINE * 0.9:
            try:
                res = minimize(
                    _objective_centers, 
                    best_centers.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                _update_best(res.x)
            except Exception:
                pass

    # Fallback if no valid result
    if best_centers is None:
        # Simple grid 4x8
        grid_4_8 = []
        for i in range(4):
            for j in range(8):
                grid_4_8.append([(i + 0.5) / 4.0, (j + 0.5) / 8.0])
        best_centers = np.array(grid_4_8)
        
    # Construct final array
    r = _radii_from_centers(best_centers)
    result = np.zeros((N, 3))
    result[:, 0] = best_centers[:, 0]
    result[:, 1] = best_centers[:, 1]
    result[:, 2] = r
    
    # Sanity check: ensure radii are not negative and circles are inside
    # _radii_from_centers ensures this by construction, but let's double check.
    # If any radius is 0, it's a degenerate circle, which is valid (disjoint interiors).
    
    return result
