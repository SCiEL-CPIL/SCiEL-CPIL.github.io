import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools
import time

N = 11
SQRT3 = np.sqrt(3)
TRI_HEIGHT = SQRT3 / 2

# Pre-compute indices for all triangles
tri_indices = np.array(list(itertools.combinations(range(N), 3)))

def get_min_area(points):
    """
    Calculate the minimum area of triangles formed by the points.
    points: (N, 2) array
    Returns: min_area (float)
    """
    # points[tri_indices[:, 0]] gives (165, 2) array of first vertices
    p1 = points[tri_indices[:, 0]]
    p2 = points[tri_indices[:, 1]]
    p3 = points[tri_indices[:, 2]]
    
    # Vectorized area calculation: 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    # Using cross product of vectors (p2-p1) and (p3-p1)
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def check_inside(points):
    """
    Check if all points are inside the triangle (0,0), (1,0), (0.5, SQRT3/2).
    Returns True if all are inside, False otherwise.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # 1. y >= 0
    if np.any(y < -1e-9):
        return False
    # 2. y <= SQRT3 * x  =>  SQRT3*x - y >= 0
    if np.any(SQRT3 * x - y < -1e-9):
        return False
    # 3. y <= -SQRT3 * (x - 1) => -SQRT3*x + SQRT3 - y >= 0
    if np.any(-SQRT3 * x + SQRT3 - y < -1e-9):
        return False
    return True

def objective(x):
    """
    Objective function for minimization.
    We want to maximize min_area, so we minimize -min_area.
    We add a penalty for points outside the triangle.
    """
    points = x.reshape(N, 2)
    
    # Calculate penalty for points outside the triangle
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Distances to boundaries (negative means outside)
    d1 = y_coords
    d2 = SQRT3 * x_coords - y_coords
    d3 = -SQRT3 * x_coords + SQRT3 - y_coords
    
    # Penalty is sum of squared negative distances
    neg_d1 = np.maximum(0, -d1)
    neg_d2 = np.maximum(0, -d2)
    neg_d3 = np.maximum(0, -d3)
    
    penalty = np.sum(neg_d1**2) + np.sum(neg_d2**2) + np.sum(neg_d3**2)
    
    # If penalty is large, return a large value immediately to guide DE away
    if penalty > 1e-4:
        return penalty * 1000.0
    
    # Calculate min area
    min_area = get_min_area(points)
    
    # We want to maximize min_area, so return negative
    return -min_area + penalty

def construct():
    start_time = time.time()
    total_time_limit = 120.0 # Safety margin for 150s limit
    
    # Bounds for DE
    # x in [0, 1], y in [0, SQRT3/2]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, TRI_HEIGHT))
    
    # Differential Evolution
    # We allocate about 60 seconds for DE
    de_time_limit = 60.0
    
    try:
        # Use a callback to stop DE if time runs out? 
        # scipy's DE doesn't have a direct time limit, but we can monitor iterations.
        # Alternatively, we can just run it and hope it finishes, or use a wrapper.
        # Since we can't easily interrupt DE mid-run in a simple script without threading,
        # we will set conservative maxiter and popsize.
        # However, DE can be slow. Let's try to run it.
        
        # If DE takes too long, it might timeout.
        # Let's run DE with a reasonable maxiter.
        # 11 points -> 22 dimensions.
        # popsize=10, maxiter=500 is a good start.
        
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=1000,
            popsize=15,
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        best_x = de_result.x
        best_val = de_result.fun
        
    except Exception as e:
        # Fallback to random if DE fails
        best_x = None
        best_val = np.inf

    # Check time
    elapsed = time.time() - start_time
    remaining_time = total_time_limit - elapsed
    
    if remaining_time < 5.0:
        # Not enough time for polish, return best from DE if valid
        if best_x is not None:
            points = best_x.reshape(N, 2)
            if check_inside(points):
                return points
            # If not inside, try to project or return best effort
            # For now, return points (might be slightly outside, evaluator might reject or penalize)
            # Better to return a known valid point if this fails
            return _fallback_point()
        
        return _fallback_point()

    # Local Polish using Nelder-Mead
    # We will run a few restarts if time permits
    
    candidates = []
    if best_x is not None:
        candidates.append(best_x)
    
    # Add a few random valid points as candidates for restart
    rng = np.random.RandomState(42)
    for _ in range(3):
        # Generate random points inside triangle
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = TRI_HEIGHT * v
            pts.append([x, y])
        candidates.append(np.array(pts).flatten())
    
    best_polished_x = None
    best_polished_val = np.inf
    
    # Allocate remaining time to polish
    # Nelder-Mead is fast. We can run it multiple times.
    
    for cand in candidates:
        if time.time() - start_time > total_time_limit - 10.0:
            break
            
        try:
            res = minimize(
                objective, 
                cand, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < best_polished_val:
                best_polished_val = res.fun
                best_polished_x = res.x
                
        except Exception:
            continue
            
    # Final check and return
    if best_polished_x is not None:
        points = best_polished_x.reshape(N, 2)
        if check_inside(points):
            return points
    
    # Fallback
    return _fallback_point()

def _fallback_point():
    # Return a simple valid configuration, e.g., grid-like or random
    # Just use the seed program's random logic to ensure validity
    import random
    rng = random.Random(7)
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.98 and u > 0.01 and v > 0.01:
            points.append([u + 0.5 * v, TRI_HEIGHT * v])
    return np.array(points)
