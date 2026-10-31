import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)

def triangle_areas_vectorized(points):
    """
    Calculate the areas of all triangles formed by the points using vectorization.
    points: shape (N, 2)
    returns: array of shape (N*(N-1)*(N-2)/6,)
    """
    n = points.shape[0]
    # Generate indices for all unique triplets (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Or using cross product magnitude
    # v1 = p2 - p1, v2 = p3 - p1
    # area = 0.5 * |v1_x * v2_y - v1_y * v2_x|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def penalty(points):
    """
    Penalty for points outside the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    x = points[:, 0]
    y = points[:, 1]
    # Inequalities:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1) => y + sqrt(3)*x - sqrt(3) <= 0
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    
    # Penalty is sum of squared violations
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    areas = triangle_areas_vectorized(points)
    min_area = np.min(areas)
    pen = penalty(points)
    # We want to maximize min_area, so minimize -min_area
    # Add large penalty for going out of bounds
    return -min_area + 100.0 * pen

def safe_objective(flat_params):
    points = flat_params.reshape(N, 2)
    # Basic sanity check to avoid numerical issues in area calc if points fly away
    if np.any(np.abs(points) > 10):
        return 1e6
    return objective(flat_params)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Use a set of diverse initializations
    # 1. Random uniform in triangle
    # 2. Grid-like distribution
    # 3. Known heuristic (e.g. vertices + centroid + others)
    
    inits = []
    
    # Random restarts
    for seed in range(100):
        if time.time() - start_time > time_limit - 10:
            break
        
        rng = np.random.RandomState(seed)
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
        inits.append(init_points)
        
    # Add a "structured" initialization: 3 corners, 3 midpoints, 1 center, 2 random
    structured = np.zeros((N, 2))
    structured[0] = [0.0, 0.0]
    structured[1] = [1.0, 0.0]
    structured[2] = [0.5, SQRT3/2]
    structured[3] = [0.5, 0.0]
    structured[4] = [0.25, SQRT3/4]
    structured[5] = [0.75, SQRT3/4]
    structured[6] = [0.5, SQRT3/6]
    # Perturb the last 4 to break symmetry
    rng = np.random.RandomState(42)
    for i in range(7, N):
        while True:
            u = rng.rand()
            v = rng.rand()
            if u + v <= 1:
                break
        w = 1 - u - v
        structured[i, 0] = v * 1.0 + w * 0.5
        structured[i, 1] = w * SQRT3 / 2.0
    inits.append(structured)
    
    for init_points in inits:
        if time.time() - start_time > time_limit - 10:
            break
        
        flat_init = init_points.flatten()
        
        try:
            # Powell is robust for non-smooth functions (min of areas)
            # Nelder-Mead is also good.
            # Let's try Powell with more iterations since it's fast now.
            result = minimize(
                safe_objective, 
                flat_init, 
                method='Powell', 
                options={'maxiter': 200, 'maxfev': 50000, 'disp': False, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            
            opt_points = result.x.reshape(N, 2)
            areas = triangle_areas_vectorized(opt_points)
            min_area = np.min(areas)
            pen = penalty(opt_points)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 100 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    if best_points is None:
        # Fallback
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
