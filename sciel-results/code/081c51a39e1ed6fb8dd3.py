import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 50

def _get_radius(points):
    """
    Calculate the maximum radius for the given points.
    points: (N, 2) array
    Returns: radius (float)
    """
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return -np.inf
        
    # Pairwise distances
    if len(points) > 1:
        dists = pdist(points)
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    # Wall distances
    # Distance to x-bounds: 0.5 - |x|
    # Distance to y-bounds: 0.5 - |y|
    x_dists = 0.5 - np.abs(points[:, 0])
    y_dists = 0.5 - np.abs(points[:, 1])
    
    min_wall_dist = np.min(np.min(x_dists), np.min(y_dists))
    
    # Radius is half the minimum distance between centers,
    # limited by the distance to the walls.
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    
    return radius

def _neg_radius(points_flat):
    """
    Objective function for minimization.
    Minimizes -radius.
    """
    points = points_flat.reshape(N, 2)
    
    # If points are far outside, return a large positive value to discourage
    # This helps Nelder-Mead stay in the valid region
    if np.any(np.abs(points) > 1.0):
        return 1000.0
        
    r = _get_radius(points)
    return -r

def construct():
    start_time = time.time()
    # Deadline: 150s limit, keep safety margin. 120s is the stated limit in prompt, 
    # but "KILLS after 150s". Let's use 130s to be safe.
    deadline = 130.0
    
    best_points = None
    best_score = -np.inf
    
    # Initial Configuration 1: 7x7 grid + 1
    # 7x7 grid has 49 points.
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    grid_pts = np.array([(x, y) for y in ticks for x in ticks])
    # Add 50th point. Place it near the center but offset to avoid overlap initially.
    # Center is (0,0).
    init1 = np.vstack([grid_pts, [[0.05, 0.05]]])
    
    # Initial Configuration 2: 5x10 grid
    ticks_x = (np.arange(5) + 0.5) / 5 - 0.5
    ticks_y = (np.arange(10) + 0.5) / 10 - 0.5
    grid_pts_50 = np.array([(x, y) for y in ticks_y for x in ticks_x])
    init2 = grid_pts_50
    
    # Initial Configuration 3: Random
    np.random.seed(123)
    init3 = np.random.uniform(-0.4, 0.4, (N, 2))
    
    inits = [init1, init2, init3]
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Check time for this specific run
        # Nelder-Mead on 100 vars can take time.
        # Let's allow ~30s per run.
        
        try:
            res = minimize(_neg_radius, init.flatten(), method='Nelder-Mead', 
                           options={
                               'maxiter': 1000, 
                               'xatol': 1e-7, 
                               'fatol': 1e-9, 
                               'adaptive': True
                           })
            
            # Check if result is better
            score = -res.fun # res.fun is -radius
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(N, 2)
                
        except Exception as e:
            # Ignore errors and try next init
            pass

    # If no optimization succeeded, fall back to init1
    if best_points is None:
        best_points = init1
        
    # Ensure points are within bounds (clip to be safe, though optimizer should keep them in)
    # Clipping might slightly reduce radius if they were on the boundary, 
    # but ensures validity.
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
