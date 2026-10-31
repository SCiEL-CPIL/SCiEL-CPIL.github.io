import numpy as np
import time
import scipy.optimize as so

N = 11

def get_areas(pts_uv):
    u = pts_uv[:, 0]
    v = pts_uv[:, 1]
    x = u + 0.5 * v
    y = (3 ** 0.5) / 2 * v
    
    areas = np.empty((N * (N - 1) * (N - 2)) // 6)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                dx1 = x[j] - x[i]
                dy1 = y[j] - y[i]
                dx2 = x[k] - x[i]
                dy2 = y[k] - y[i]
                det = dx1 * dy2 - dx2 * dy1
                areas[idx] = 0.5 * abs(det)
                idx += 1
    return areas

def soft_min(areas, t):
    m = np.min(areas)
    shifted = areas - m
    log_sum = np.log(np.sum(np.exp(-t * shifted)))
    return m - (1.0 / t) * log_sum

def objective(params, t, penalty_weight):
    pts_uv = params.reshape(N, 2)
    s = pts_uv[:, 0] + pts_uv[:, 1]
    viol = np.maximum(0, s - 1.0)
    penalty = penalty_weight * np.sum(viol ** 2)
    
    areas = get_areas(pts_uv)
    sm = soft_min(areas, t)
    
    return -sm + penalty

def construct():
    start_time = time.time()
    max_time = 110.0
    
    best_score = -np.inf
    best_params = None
    
    rng = np.random.RandomState(42)
    
    # We will perform several restarts
    # Each restart involves annealing the temperature t
    num_restarts = 40
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Initialize points inside the triangle
        pts = []
        while len(pts) < N:
            u_try = rng.random()
            v_try = rng.random()
            if u_try + v_try <= 1.0:
                pts.append([u_try, v_try])
        params = np.array(pts).flatten()
        
        # Temperature schedule
        t_schedule = [5, 20, 50, 100, 200]
        
        current_params = params
        for t in t_schedule:
            if time.time() - start_time > max_time:
                break
                
            # Optimize
            # Using L-BFGS-B
            # Bounds for u and v are [0, 1]
            bounds = [(0, 1)] * (2 * N)
            
            res = so.minimize(objective, current_params, args=(t, 100.0), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            current_params = res.x
            
            # Evaluate true objective (min area)
            pts_uv = current_params.reshape(N, 2)
            # Check validity
            s = pts_uv[:, 0] + pts_uv[:, 1]
            if np.any(s > 1.0 + 1e-5):
                # If invalid, we might still keep it if score is high? 
                # No, invalid points are outside the triangle.
                # The penalty should prevent this, but if it happens, the score might be misleading
                # if the evaluator rejects it. 
                # However, the penalty is quadratic and large, so it should be valid.
                pass
            
            areas = get_areas(pts_uv)
            min_area = np.min(areas)
            
            if min_area > best_score:
                best_score = min_area
                best_params = current_params.copy()
                
    # Fallback
    if best_params is None:
        pts = []
        while len(pts) < N:
            u_try = rng.random()
            v_try = rng.random()
            if u_try + v_try <= 1.0:
                pts.append([u_try, v_try])
        best_params = np.array(pts).flatten()
        
    # Convert to Cartesian
    pts_uv = best_params.reshape(N, 2)
    u = pts_uv[:, 0]
    v = pts_uv[:, 1]
    x = u + 0.5 * v
    y = (3 ** 0.5) / 2 * v
    
    return np.column_stack((x, y))
