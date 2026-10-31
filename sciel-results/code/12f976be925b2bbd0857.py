import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
TOTAL_AREA = SQRT3 / 4.0

def triangle_areas(points):
    """
    Calculate the areas of all triangles formed by the points.
    """
    n = points.shape[0]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(
                    p1[0] * (p2[1] - p3[1]) + 
                    p2[0] * (p3[1] - p1[1]) + 
                    p3[0] * (p1[1] - p2[1])
                )
                areas.append(area)
    return np.array(areas)

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def soft_min(areas, beta):
    """
    Smooth approximation of the minimum.
    """
    # To avoid overflow in exp, subtract max
    max_a = np.max(areas)
    # We want -min, so we use logsumexp of -areas
    # -min(a) = - ( - logsumexp(-a) / beta ) ? No.
    # min(a) ~ -1/beta * log(sum(exp(-beta * a)))
    # So -min(a) ~ 1/beta * log(sum(exp(-beta * a)))
    # Let's compute logsumexp(-beta * areas)
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return log_sum_exp / beta

def objective_smooth(flat_params, beta):
    points = flat_params.reshape(N, 2)
    areas = triangle_areas(points)
    # If any area is 0, soft_min is 0. 
    # We want to maximize min area, so minimize -soft_min.
    val = -soft_min(areas, beta)
    pen = penalty(points)
    return val + 100.0 * pen

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Initial guesses:
    # 1. Vertices + Midpoints + Centroid + 4 random
    # 2. Grid
    # 3. Random
    
    def get_initial_points(seed):
        rng = np.random.RandomState(seed)
        points = np.zeros((N, 2))
        
        # Fixed structural points
        # Vertices
        points[0] = [0.0, 0.0]
        points[1] = [1.0, 0.0]
        points[2] = [0.5, SQRT3/2]
        
        # Midpoints
        points[3] = [0.5, 0.0]
        points[4] = [0.25, SQRT3/4]
        points[5] = [0.75, SQRT3/4]
        
        # Centroid
        points[6] = [0.5, SQRT3/6]
        
        # Remaining 4 points: random inside
        for i in range(7, N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            points[i, 0] = v * 1.0 + w * 0.5
            points[i, 1] = w * SQRT3 / 2.0
            
        return points

    # Multi-stage beta annealing
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    current_points = None
    
    for seed in range(30):
        if time.time() - start_time > time_limit:
            break
            
        init_points = get_initial_points(seed)
        
        # Annealing loop
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            try:
                result = minimize(
                    objective_smooth,
                    init_points.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfev': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                init_points = result.x.reshape(N, 2)
            except Exception:
                pass
        
        # Evaluate final score
        if init_points is not None:
            min_area = np.min(triangle_areas(init_points))
            pen = penalty(init_points)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 100 * pen
            
            if score > best_score:
                best_score = score
                best_points = init_points.copy()
                
    if best_points is None:
        # Fallback
        return get_initial_points(0)
        
    return best_points
