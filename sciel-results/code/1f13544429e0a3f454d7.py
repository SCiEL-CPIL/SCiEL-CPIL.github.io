import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

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

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    min_area = np.min(triangle_areas(points))
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def safe_objective(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6
    return objective(flat_params)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple random restarts with Powell
    for seed in range(50):
        if time.time() - start_time > time_limit:
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
        
        flat_init = init_points.flatten()
        
        try:
            result = minimize(
                safe_objective, 
                flat_init, 
                method='Powell', 
                options={'maxiter': 100, 'maxfev': 10000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            min_area = np.min(triangle_areas(opt_points))
            pen = penalty(opt_points)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            bh_result = basinhopping(
                safe_objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 50, 'maxfev': 5000},
                niter=10,
                seed=123
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = np.min(triangle_areas(bh_points))
            bh_pen = penalty(bh_points)
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
