import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def is_valid(params):
        W = params[-1]
        H = 2.0 - W
        if W <= 0.01 or H <= 0.01:
            return False
        r = params[0::3][:N]
        x = params[1::3][:N]
        y = params[2::3][:N]
        
        # Boundary check
        for i in range(N):
            if x[i] < r[i] - 1e-6 or x[i] + r[i] > W + 1e-6 or y[i] < r[i] - 1e-6 or y[i] + r[i] > H + 1e-6:
                return False
                
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist2 = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist2 < min_dist*min_dist - 1e-6:
                    return False
        return True

    def get_score(params):
        r = params[0::3][:N]
        return np.sum(r)

    # Initial guess: Grid
    side = 5
    W0 = 1.0
    H0 = 1.0
    x0 = np.array([(i % side + 0.5) / side for i in range(N)])
    y0 = np.array([(i // side + 0.5) / side for i in range(N)])
    r0 = np.full(N, 0.099)
    
    params0 = np.zeros(N * 3 + 1)
    params0[0::3] = r0
    params0[1::3] = x0
    params0[2::3] = y0
    params0[-1] = W0
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, 1.0))  # r
        bounds.append((0.01, 1.9))  # x
        bounds.append((0.01, 1.9))  # y
    bounds.append((0.1, 1.9))  # W
        
    def objective(params):
        W = params[-1]
        H = 2.0 - W
        if W <= 0.01 or H <= 0.01:
            return 1e10
            
        r = params[0::3][:N]
        x = params[1::3][:N]
        y = params[2::3][:N]
        
        penalty = 0.0
        
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] + r[i] > W:
                penalty += (x[i] + r[i] - W)**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] + r[i] > H:
                penalty += (y[i] + r[i] - H)**2
                
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist2 = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist2 < min_dist*min_dist:
                    dist = np.sqrt(dist2)
                    overlap = min_dist - dist
                    penalty += overlap**2
                    
        sum_r = np.sum(r)
        return -sum_r + 50.0 * penalty

    best_params = params0.copy()
    best_score = get_score(best_params)
    
    num_restarts = 10
    max_time_per_restart = 10.0
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        p = params0.copy()
        if k > 0:
            # Random jitter
            jitter = np.random.normal(0, 0.05, N*3)
            p[np.arange(N*3)] += jitter
            p[-1] = 1.0 + np.random.normal(0, 0.2)
            # Clamp to bounds
            p = np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds])
            
        # Optimize
        try:
            res = minimize(objective, p, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Check if result is valid
            if is_valid(res.x):
                score = get_score(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # Final check
    if not is_valid(best_params):
        # Fallback to grid
        best_params = params0.copy()
        
    r = best_params[0::3][:N]
    x = best_params[1::3][:N]
    y = best_params[2::3][:N]
    
    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    
    return result
