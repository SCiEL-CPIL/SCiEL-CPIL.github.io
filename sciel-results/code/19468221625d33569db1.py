import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
# Vertices of the equilateral triangle
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])

def get_areas(points):
    """
    Vectorized calculation of areas of all triangles formed by N points.
    points: (N, 2) array
    Returns: array of shape (N*(N-1)*(N-2)/6, )
    """
    n = points.shape[0]
    # Generate indices
    idx = np.indices((n, n, n)).reshape(3, -1).T
    # Filter i < j < k
    mask = (idx[:, 0] < idx[:, 1]) & (idx[:, 1] < idx[:, 2])
    idx = idx[mask]
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    # cross product in 2D: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def penalty(points):
    """
    Penalty for points outside the triangle.
    Triangle inequalities:
    1. y >= 0
    2. sqrt(3)*x - y >= 0
    3. sqrt(3) - sqrt(3)*x - y >= 0
    """
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    # Sum of squared violations
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params, T, pen_weight):
    """
    Objective function to minimize.
    -softmin(areas) + pen_weight * penalty
    """
    points = flat_params.reshape(N, 2)
    areas = get_areas(points)
    # Softmin: -1/T * log(sum(exp(-T * areas)))
    # To avoid overflow, subtract max
    max_area = np.max(areas)
    log_sum_exp = max_area + np.log(np.sum(np.exp(-T * (areas - max_area))))
    soft_min_val = -log_sum_exp / T
    
    pen = penalty(points)
    return -soft_min_val + pen_weight * pen

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Annealing schedule for T (temperature of softmin)
    # Low T -> smooth, High T -> sharp (closer to min)
    T_schedule = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    pen_weight = 100.0
    
    # Number of restarts
    n_restarts = 40
    
    for seed in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        rng = np.random.RandomState(seed)
        
        # Initialize points inside the triangle
        # Using barycentric coords u, v, w
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
        
        current_params = init_points.flatten()
        
        # Optimize with increasing T
        for T in T_schedule:
            if time.time() - start_time > time_limit:
                break
            
            # Use L-BFGS-B for gradient-based optimization
            # If L-BFGS-B fails or is slow, fall back to Powell
            try:
                res = minimize(
                    objective, 
                    current_params, 
                    args=(T, pen_weight),
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'maxls': 50}
                )
                current_params = res.x
            except Exception:
                # Fallback to Powell if L-BFGS-B has issues
                try:
                    res = minimize(
                        objective, 
                        current_params, 
                        args=(T, pen_weight),
                        method='Powell', 
                        options={'maxiter': 100, 'maxfev': 5000}
                    )
                    current_params = res.x
                except Exception:
                    continue
        
        # Evaluate the final result
        opt_points = current_params.reshape(N, 2)
        areas = get_areas(opt_points)
        min_area = np.min(areas)
        pen = penalty(opt_points)
        
        if pen < 1e-4:
            score = min_area
        else:
            score = min_area - 1000 * pen
            
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()

    if best_points is None:
        # Fallback to centroid
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
