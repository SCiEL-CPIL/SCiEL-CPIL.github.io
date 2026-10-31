import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Variables: W, x[0..N-1], y[0..N-1], r[0..N-1]
    # H = 2 - W
    # Total vars: 1 + 3*N = 64
    
    def unpack(v):
        W = v[0]
        H = 2.0 - W
        x = v[1:1+N]
        y = v[1+N:1+2*N]
        r = v[1+2*N:1+3*N]
        return W, H, x, y, r

    def objective_and_penalty(v, lambda_pen):
        W, H, x, y, r = unpack(v)
        
        # If W or H is negative or too small, huge penalty
        if W <= 0 or H <= 0:
            return 1e10, 1e10
            
        sum_r = np.sum(r)
        
        # Boundary penalties
        # r_i <= x_i, r_i <= W-x_i, r_i <= y_i, r_i <= H-y_i
        pen_x1 = np.sum(np.maximum(0, r - x)**2)
        pen_x2 = np.sum(np.maximum(0, r - (W - x))**2)
        pen_y1 = np.sum(np.maximum(0, r - y)**2)
        pen_y2 = np.sum(np.maximum(0, r - (H - y))**2)
        boundary_pen = pen_x1 + pen_x2 + pen_y1 + pen_y2
        
        # Overlap penalties
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r_ij = r[i] + r[j]
                if dist_sq < sum_r_ij**2:
                    # dist < sum_r
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                    violation = sum_r_ij - dist
                    overlap_pen += violation**2
        
        total_pen = boundary_pen + overlap_pen
        obj = -sum_r + lambda_pen * total_pen
        return obj, total_pen

    def run_optimization(init_v, max_iter=200):
        # Start with small lambda, increase
        lambda_pen = 1.0
        current_v = init_v.copy()
        
        for _ in range(5): # Few penalty iterations
            def func(v):
                obj, _ = objective_and_penalty(v, lambda_pen)
                return obj
            
            # Bounds:
            # W in [0.2, 1.8] (since W+H=2, symmetric)
            # x, y, r > 0
            bounds = [(0.2, 1.8)] # W
            for _ in range(N):
                bounds.append((1e-6, 1.0)) # x
            for _ in range(N):
                bounds.append((1e-6, 1.0)) # y
            for _ in range(N):
                bounds.append((1e-6, 1.0)) # r
                
            res = minimize(func, current_v, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-8})
            current_v = res.x
            
            _, pen = objective_and_penalty(current_v, lambda_pen)
            if pen < 1e-6:
                break
            lambda_pen *= 10.0
            
        return current_v

    def get_score(v):
        W, H, x, y, r = unpack(v)
        if W <= 0 or H <= 0:
            return -1e10
        # Check validity
        for i in range(N):
            if r[i] < 1e-9:
                return -1e10
            if r[i] > x[i] + 1e-9 or r[i] > W - x[i] + 1e-9:
                return -1e10
            if r[i] > y[i] + 1e-9 or r[i] > H - y[i] + 1e-9:
                return -1e10
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -1e10
        return np.sum(r)

    best_score = 0.0
    best_v = None
    
    # Initial guesses
    inits = []
    
    # 1. Grid
    W_init = 1.0
    H_init = 1.0
    side = int(np.ceil(np.sqrt(N)))
    x_grid = np.zeros(N)
    y_grid = np.zeros(N)
    r_grid = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        x_grid[i] = (col + 0.5) * (W_init / side)
        y_grid[i] = (row + 0.5) * (H_init / side)
        r_grid[i] = 0.4 * (W_init / side)
    v_grid = np.concatenate(([W_init], x_grid, y_grid, r_grid))
    inits.append(v_grid)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(10):
        W_rand = np.random.uniform(0.5, 1.5)
        H_rand = 2.0 - W_rand
        x_rand = np.random.uniform(0.1, W_rand-0.1, N)
        y_rand = np.random.uniform(0.1, H_rand-0.1, N)
        r_rand = np.random.uniform(0.05, 0.15, N)
        v_rand = np.concatenate(([W_rand], x_rand, y_rand, r_rand))
        inits.append(v_rand)
        
    # 3. Hexagonal-ish
    W_hex = 1.0
    H_hex = 1.0
    x_hex = np.zeros(N)
    y_hex = np.zeros(N)
    r_hex = np.zeros(N)
    rows = 4
    cols = 5
    idx = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= N: break
            x_pos = (c_idx + 0.5) * (W_hex / cols)
            if r_idx % 2 == 1:
                x_pos += 0.5 * (W_hex / cols)
            y_pos = (r_idx + 0.5) * (H_hex / rows)
            x_hex[idx] = x_pos
            y_hex[idx] = y_pos
            r_hex[idx] = 0.4 * (W_hex / cols)
            idx += 1
    v_hex = np.concatenate(([W_hex], x_hex, y_hex, r_hex))
    inits.append(v_hex)

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            v_opt = run_optimization(init)
            score = get_score(v_opt)
            if score > best_score:
                best_score = score
                best_v = v_opt
        except Exception:
            continue

    # If no valid solution found, return a safe default
    if best_v is None:
        # Return the grid one, scaled down slightly to ensure validity
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.45 / side
            cells.append([x, y, r])
        return np.array(cells)

    W, H, x, y, r = unpack(best_v)
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
