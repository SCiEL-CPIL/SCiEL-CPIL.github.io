import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_val = -np.inf
    best_result = None
    
    # Variables: x[0..20], y[0..20], r[0..20], W
    # H = 2 - W
    
    def unpack(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        W = params[3*N]
        H = 2.0 - W
        return x, y, r, W, H

    def objective(params):
        r = params[2*N:3*N]
        return -np.sum(r)

    def constraints(params):
        x, y, r, W, H = unpack(params)
        cons = []
        
        # Boundary constraints: x_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*N] - p[i] - p[2*N+i]}) # x_i <= W - r_i
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]}) # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[3*N]) - p[N+i] - p[2*N+i]}) # y_i <= H - r_i
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i] - 1e-5}) # r_i > 0
            
        # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda p, i=i, j=j: (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2 - (p[2*N+i]+p[2*N+j])**2
                })
                
        return cons

    def run_optimization(x0, y0, r0, W0, maxiter=200):
        params0 = np.concatenate([x0, y0, r0, [W0]])
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((1e-5, 1.0)) # r
        bounds.append((0.5, 1.5)) # W
        
        # SLSQP is slow, so we limit iterations
        try:
            res = minimize(objective, params0, method='SLSQP', bounds=bounds, 
                           constraints=constraints(params0),
                           options={'maxiter': maxiter, 'ftol': 1e-8})
            
            if not res.success:
                # Even if not successful, check if the result is valid
                pass
                
            x, y, r, W, H = unpack(res.x)
            
            # Check validity strictly
            valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-6 or x[i] > W - r[i] + 1e-6:
                    valid = False
                if y[i] < r[i] - 1e-6 or y[i] > H - r[i] + 1e-6:
                    valid = False
                if r[i] <= 1e-6:
                    valid = False
                if W < 0.5 - 1e-6 or W > 1.5 + 1e-6:
                    valid = False
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r)
                return (x, y, r, W, H), val
            else:
                return None, -np.inf
                
        except Exception:
            return None, -np.inf

    def generate_grid_config(rows, cols, W_init):
        H_init = 2.0 - W_init
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Calculate max radius for this grid
        # We want to fit 'rows' by 'cols' circles.
        # If we have exactly rows*cols, r = min(W/cols, H/rows)/2
        # If we have fewer, we can space them out more.
        
        total_cells = rows * cols
        if total_cells < N:
             return None
             
        # We pick N positions from the grid
        # Simple strategy: fill row by row
        positions = []
        for i in range(rows):
            for j in range(cols):
                positions.append((i, j))
                if len(positions) == N:
                    break
            if len(positions) == N:
                break
        
        # Calculate base radius
        # To maximize sum of radii, we want them as large as possible.
        # In a grid, limiting factor is usually the cell size.
        # Let's estimate r as half the minimum cell dimension.
        cell_w = W_init / cols
        cell_h = H_init / rows
        r_init = 0.5 * min(cell_w, cell_h) * 0.95 # 5% margin
        
        for k, (i, j) in enumerate(positions):
            x0[k] = (j + 0.5) * cell_w
            y0[k] = (i + 0.5) * cell_h
            r0[k] = r_init
            
        return x0, y0, r0, W_init

    configs = []
    
    # 1. 4x6 Grid (24 spots, pick 21)
    for W_init in [0.8, 1.0, 1.2]:
        cfg = generate_grid_config(4, 6, W_init)
        if cfg: configs.append(cfg)
        
    # 2. 5x5 Grid (25 spots, pick 21)
    for W_init in [0.9, 1.0, 1.1]:
        cfg = generate_grid_config(5, 5, W_init)
        if cfg: configs.append(cfg)
        
    # 3. 3x7 Grid (21 spots exactly)
    for W_init in [0.7, 1.0, 1.3]:
        cfg = generate_grid_config(3, 7, W_init)
        if cfg: configs.append(cfg)
        
    # 4. 7x3 Grid
    for W_init in [0.7, 1.0, 1.3]:
        cfg = generate_grid_config(7, 3, W_init)
        if cfg: configs.append(cfg)

    # 5. Random configurations
    np.random.seed(42)
    for _ in range(5):
        W_init = np.random.uniform(0.8, 1.2)
        H_init = 2.0 - W_init
        x0 = np.random.uniform(0.1, W_init-0.1, N)
        y0 = np.random.uniform(0.1, H_init-0.1, N)
        # Start with small radii to avoid immediate overlap penalties
        r0 = np.full(N, 0.05) 
        configs.append((x0, y0, r0, W_init))

    # Run optimization for each config
    for i, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        x0, y0, r0, W0 = cfg
        try:
            res, val = run_optimization(x0, y0, r0, W0, maxiter=100)
            if val > best_val:
                best_val = val
                best_result = res
        except Exception:
            pass

    # Polish phase: if we have a valid result, try to improve it
    if best_result is not None and time.time() < deadline - 5.0:
        x, y, r, W, H = best_result
        try:
            res_polish, val_polish = run_optimization(x, y, r, W, maxiter=200)
            if val_polish > best_val:
                best_val = val_polish
                best_result = res_polish
        except Exception:
            pass

    if best_result is None:
        # Fallback: simple grid
        side = 5
        W = 1.0
        H = 1.0
        r = 0.1
        out = np.zeros((N, 3))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                out[idx, 0] = (j + 0.5) * (W / side)
                out[idx, 1] = (i + 0.5) * (H / side)
                out[idx, 2] = r
                idx += 1
        return out

    x, y, r, W, H = best_result
    out = np.column_stack((x, y, r))
    return out

