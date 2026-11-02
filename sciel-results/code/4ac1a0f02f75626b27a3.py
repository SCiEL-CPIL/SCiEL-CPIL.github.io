import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_solution = None
    best_score = -np.inf
    
    def penalty_objective(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x in [r, 1-r], y in [r, 1-r]
        viol_l = np.maximum(0, r - x)
        viol_b = np.maximum(0, r - y)
        viol_r = np.maximum(0, x - (1 - r))
        viol_t = np.maximum(0, y - (1 - r))
        
        pen_bound = np.sum(viol_l**2) + np.sum(viol_b**2) + np.sum(viol_r**2) + np.sum(viol_t**2)
        
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = r[i] + r[j]
                viol = np.maximum(0, min_dist - dist)
                pen_overlap += viol**2
        
        mu = 500.0
        return -sum_r + mu * (pen_bound + pen_overlap)

    def run_optimization(init_params, max_iters=1000):
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
        for i in range(N):
            bounds.append((0.0, 1.0)) # y
        for i in range(N):
            bounds.append((1e-6, 0.5)) # r
            
        result = minimize(penalty_objective, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iters, 'ftol': 1e-12})
        return result

    def is_valid(x, y, r, tol=1e-5):
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol or y[i] < r[i] - tol or y[i] > 1 - r[i] + tol or r[i] <= 0:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    # Initial Configurations
    
    # 1. Grid
    side = 5
    grid_x = []
    grid_y = []
    grid_r = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.5 / side * 0.95
            grid_x.append(cx)
            grid_y.append(cy)
            grid_r.append(cr)
            count += 1
        if count >= N:
            break
            
    init_grid = np.array(grid_x + grid_y + grid_r)
    
    # 2. Random starts
    inits = [init_grid]
    
    for seed in range(1, 20):
        if time.time() - start_time > time_limit - 10:
            break
        np.random.seed(seed)
        # Random positions, small radii
        x0 = np.random.rand(N) * 0.8 + 0.1
        y0 = np.random.rand(N) * 0.8 + 0.1
        r0 = np.full(N, 0.05)
        init_rand = np.concatenate([x0, y0, r0])
        inits.append(init_rand)
        
    for idx, init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = run_optimization(init, max_iters=200)
            
            x = res.x[:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            
            if is_valid(x, y, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_solution = np.column_stack((x, y, r))
                    
        except Exception as e:
            pass

    if best_solution is None:
        # Fallback to grid
        return np.array([[ (i+0.5)/5, (j+0.5)/5, 0.5/5-1e-3 ] for i in range(5) for j in range(5)][:N])
        
    return best_solution
