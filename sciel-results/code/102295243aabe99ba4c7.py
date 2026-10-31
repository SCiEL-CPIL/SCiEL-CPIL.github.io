import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    max_time = 140.0  # Safety margin
    
    best_result = None
    best_score = -np.inf

    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    def boundary_constraints(params):
        vals = np.zeros(4*N)
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            vals[4*i] = x - r
            vals[4*i+1] = 1 - r - x
            vals[4*i+2] = y - r
            vals[4*i+3] = 1 - r - y
        return vals

    def overlap_constraints(params):
        vals = np.zeros(N*(N-1)//2)
        idx = 0
        for i in range(N):
            xi = params[3*i]
            yi = params[3*i+1]
            ri = params[3*i+2]
            for j in range(i+1, N):
                xj = params[3*j]
                yj = params[3*j+1]
                rj = params[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                vals[idx] = dist_sq - sum_r**2
                idx += 1
        return vals

    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    def is_valid(params):
        for i in range(N):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            if x < r - 1e-6 or x > 1 - r + 1e-6:
                return False
            if y < r - 1e-6 or y > 1 - r + 1e-6:
                return False
            if r < 1e-6:
                return False
        for i in range(N):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i+1, N):
                xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
                if (xi-xj)**2 + (yi-yj)**2 < (ri+rj)**2 - 1e-6:
                    return False
        return True

    # Initializations
    inits = []
    
    # 1. Grid
    rows, cols = 5, 6
    xs = np.linspace(0.5/cols, 1-0.5/cols, cols)
    ys = np.linspace(0.5/rows, 1-0.5/rows, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    r_init = 0.5/cols - 0.02
    
    init_params = []
    for i in range(N):
        x = grid_x[i]
        y = grid_y[i]
        r = r_init
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        init_params.extend([x, y, r])
    inits.append(np.array(init_params))
    
    # 2. Grid with jitter
    for _ in range(3):
        init_params = []
        for i in range(N):
            x = grid_x[i] + np.random.uniform(-0.03, 0.03)
            y = grid_y[i] + np.random.uniform(-0.03, 0.03)
            r = r_init
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            init_params.extend([x, y, r])
        inits.append(np.array(init_params))

    # 3. Random
    for _ in range(5):
        x_rand = np.random.uniform(0.2, 0.8, N)
        y_rand = np.random.uniform(0.2, 0.8, N)
        r_rand = np.full(N, 0.05)
        init_rand = []
        for i in range(N):
            x = np.clip(x_rand[i], r_rand[i], 1-r_rand[i])
            y = np.clip(y_rand[i], r_rand[i], 1-r_rand[i])
            init_rand.extend([x, y, r_rand[i]])
        inits.append(np.array(init_rand))

    # Run optimization
    for init in inits:
        if time.time() - start_time > max_time:
            break
            
        try:
            res = minimize(
                objective, 
                init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-9}
            )
            
            params = res.x
            if is_valid(params):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_result = params
                    
        except Exception as e:
            pass

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells[:N])
        
    result_array = np.zeros((N, 3))
    for i in range(N):
        result_array[i] = best_result[3*i:3*i+3]
        
    return result_array
