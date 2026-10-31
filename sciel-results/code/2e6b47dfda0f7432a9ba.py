import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all 165 triangles
TRI_INDICES = list(itertools.combinations(range(N), 3))
I_IDX = np.array([t[0] for t in TRI_INDICES])
J_IDX = np.array([t[1] for t in TRI_INDICES])
K_IDX = np.array([t[2] for t in TRI_INDICES])

def get_min_area(points):
    """
    Vectorized calculation of the minimum triangle area.
    points: (N, 2) array
    """
    p1 = points[I_IDX]
    p2 = points[J_IDX]
    p3 = points[K_IDX]
    
    # Area = 0.5 * | x1(y2-y3) + x2(y3-y1) + x3(y1-y2) |
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return np.min(areas)

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    # Constraints for triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    
    # Fast rejection for points far outside
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 1.5) or np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 1.5):
        return 1e6
    
    min_area = get_min_area(points)
    pen = penalty(points)
    
    return -min_area + 1000.0 * pen

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Bounds for Differential Evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0.0, 1.0), (0.0, SQRT3/2.0)] * N
    
    best_result = None
    best_score = np.inf # We are minimizing the objective
    
    # 1. Global Search with Differential Evolution
    try:
        # popsize=10, maxiter=100 -> ~1000-2000 evaluations
        # This is safe for 120s with vectorized objective
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=150, 
            popsize=10,
            tol=1e-6,
            disp=False,
            polish=False # We will polish manually to control time
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
            
    except Exception:
        pass

    # 2. Local Polish with Powell
    if best_result is not None and time.time() - start_time < time_limit - 5:
        try:
            polish_result = minimize(
                objective, 
                best_result, 
                method='Powell',
                options={'maxiter': 200, 'maxfev': 10000, 'disp': False}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_result = polish_result.x
        except Exception:
            pass
            
    # 3. Fallback: Multi-start Powell if DE failed or we have lots of time
    if (best_result is None or time.time() - start_time < time_limit - 20) and time.time() - start_time < time_limit - 5:
        # Try a few random restarts with Powell
        for seed in range(20):
            if time.time() - start_time > time_limit - 2:
                break
            
            rng = np.random.RandomState(seed)
            init = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u = rng.rand()
                    v = rng.rand()
                    if u + v <= 1:
                        break
                w = 1 - u - v
                init[i, 0] = v * 1.0 + w * 0.5
                init[i, 1] = w * SQRT3 / 2.0
            
            try:
                res = minimize(
                    objective, 
                    init.flatten(), 
                    method='Powell', 
                    options={'maxiter': 100, 'maxfev': 5000}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
            except Exception:
                pass

    if best_result is None:
        # Return a valid default configuration
        # Center of triangle
        return np.array([[0.5, SQRT3/6.0]] * N)
        
    points = best_result.reshape(N, 2)
    return points
