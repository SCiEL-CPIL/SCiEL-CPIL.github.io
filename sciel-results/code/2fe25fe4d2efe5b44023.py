import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def _compute_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    # Check for degeneracy
    if np.allclose(points, points[0]):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, .volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-9:
        return 0.0
        
    # Calculate all triangle areas
    min_area = float('inf')
    # Vectorized calculation of triangle areas
    # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
    # We can compute this for all pairs i, j, k with i < j < k
    
    # To be efficient, we can use broadcasting, but with N=13, 286 triangles is small.
    # However, doing it in a loop is safer and fast enough for N=13.
    
    # Alternative: Vectorized
    # indices = list(itertools.combinations(range(N), 3))
    # This creates 286 tuples.
    
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Cross product magnitude / 2
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    
    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def _negative_score(points_flat):
    """
    Objective function for minimizers.
    Reshapes flat array to (N, 2) points.
    Returns negative score to maximize score.
    """
    points = points_flat.reshape(N, 2)
    score = _compute_score(points)
    return -score

def construct():
    start_time = time.time()
    timeout = 120  # Safety margin
    
    # Initial guess: Regular N-gon
    angles = 2 * np.pi * np.arange(N) / N
    initial_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = _compute_score(initial_points)
    best_points = initial_points.copy()
    
    # Differential Evolution
    # Bounds: [-1.5, 1.5] for x and y
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    # We want to maximize, so minimize negative score
    # DE can be slow, so we limit generations and use a time check if possible,
    # but DE doesn't support early stopping easily in scipy.
    # We'll run a moderate number of generations.
    
    # To ensure we don't time out, we'll run DE with a fixed budget
    # and then polish.
    
    # Strategy:
    # 1. Run DE with maxiter=50, popsize=15 (approx 750 evaluations)
    #    Each evaluation is fast (286 triangles).
    #    750 * 286 ops is trivial.
    # 2. Take the best result and run Nelder-Mead.
    
    try:
        de_result = differential_evolution(
            _negative_score, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-8,
            init='sobol'
        )
        
        if de_result.fun < -best_score:
            best_points = de_result.x.reshape(N, 2)
            best_score = -de_result.fun
            
    except Exception as e:
        pass # If DE fails, keep initial
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
        # Polish with Nelder-Mead
        try:
            nm_result = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if nm_result.fun < -best_score:
                best_points = nm_result.x.reshape(N, 2)
                best_score = -nm_result.fun
                
        except Exception as e:
            pass
            
    # Final check and return
    # Ensure the points are reasonable (not NaN, etc.)
    if not np.all(np.isfinite(best_points)):
        return initial_points
        
    return best_points
