import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        # Check for degenerate points
        if len(points) < 3:
            return 0.0
        
        # Compute Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Compute Min Triangle Area
        # Number of triangles C(13, 3) = 286
        # Vectorized calculation
        idx = np.array(list(itertools.combinations(range(N), 3)))
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    init_points = np.column_stack((x, y))
    
    # Try a few different starting points and optimization runs
    # 1. Regular polygon
    # 2. Regular polygon with small noise
    # 3. Slightly irregular polygon
    
    starting_points = []
    starting_points.append(init_points.copy())
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise = rng.normal(0, 0.05, (N, 2))
        starting_points.append(init_points + noise)
        
    for i, start in enumerate(starting_points):
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness better
            res = minimize(
                neg_score, 
                start.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.success or (res.fun < 0 and -res.fun > best_score):
                cur_score = -res.fun
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception as e:
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
    # If no optimization worked, return the regular polygon
    if best_points is None:
        best_points = init_points
    else:
        # Verify the best point is valid
        final_score = score(best_points)
        if final_score <= 0:
             best_points = init_points

    return best_points
