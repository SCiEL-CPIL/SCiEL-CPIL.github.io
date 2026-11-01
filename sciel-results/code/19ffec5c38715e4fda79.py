import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# EVOLVE-BLOCK-START
import random

def calc_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        # Calculate convex hull area
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        if hull_area < 1e-10:
            return 0.0
            
        # Calculate all triangle areas
        # Vectorized approach
        # i, j, k
        # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
        
        n = len(points)
        min_area = float('inf')
        
        # Precompute indices for efficiency? 
        # For N=14, 364 triangles is small. Looping is fine.
        # But vectorizing the cross product is faster.
        
        # Let's stick to a simple loop for clarity and correctness, 
        # it's fast enough for N=14.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    v1 = points[j] - points[i]
                    v2 = points[k] - points[i]
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
                        
        return min_area / hull_area
    except Exception:
        return 0.0

def objective(points_flat):
    """
    Objective function for minimization.
    Minimizes negative score.
    """
    points = points_flat.reshape(-1, 2)
    score = calc_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Regular Polygon Initialization
    # Points on a circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radius = 1.0
    init_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Strategy 2: Random points on circle
    rng = np.random.RandomState(42)
    for seed in range(5):
        if time.time() > deadline:
            break
            
        if seed == 0:
            # Regular polygon
            current_init = init_points.copy()
        else:
            # Perturbed regular polygon or random on circle
            # Let's try random angles on circle
            random_angles = np.sort(rng.uniform(0, 2*np.pi, N))
            current_init = np.column_stack([np.cos(random_angles), np.sin(random_angles)])
        
        # Optimization
        # Nelder-Mead is good for small N and non-smooth functions
        try:
            res = scipy.optimize.minimize(
                objective, 
                current_init.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True}
            )
            
            optimized_points = res.x.reshape(-1, 2)
            score = calc_score(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
        except Exception:
            pass
            
    # If no points found or score is 0, return a regular polygon as fallback
    if best_points is None:
        best_points = init_points
    
    return best_points

# EVOLVE-BLOCK-END
