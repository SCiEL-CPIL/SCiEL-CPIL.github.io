import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def objective(vars):
        # vars: x1, y1, r1, x2, y2, r2, ...
        # Maximize sum of radii -> Minimize -sum(r)
        radii = vars[2::3]
        return -np.sum(radii)

    def constraints_func(vars):
        constraints = []
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Boundary constraints
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda i=i, x=x, y=y, r=r: x[i] - r[i]})
            constraints.append({'type': 'ineq', 'fun': lambda i=i, x=x, y=y, r=r: y[i] - r[i]})
            constraints.append({'type': 'ineq', 'fun': lambda i=i, x=x, y=y, r=r: 1 - x[i] - r[i]})
            constraints.append({'type': 'ineq', 'fun': lambda i=i, x=x, y=y, r=r: 1 - y[i] - r[i]})
            
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda i=i, j=j, x=x, y=y, r=r: (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2
                })
                
        return constraints

    bounds = [(0, 1), (0, 1), (0, 0.5)] * N  # x, y, r bounds. r max 0.5 is loose.

    best_score = -np.inf
    best_vars = None

    # Restart 1: Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    init_vars = []
    for i in range(N):
        col = i % side
        row = i // side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        r = 0.1 # Small initial radius
        init_vars.extend([x, y, r])
    
    if time.time() < deadline:
        res = minimize(objective, init_vars, method='SLSQP', bounds=bounds, constraints=constraints_func, options={'maxiter': 500, 'ftol': 1e-8})
        if res.success:
            best_score = -res.fun
            best_vars = res.x

    # Restart 2-10: Random initialization
    np.random.seed(42) # For reproducibility in evolution, but random restarts need variety? 
    # Actually, using a fixed seed for the whole run is safer for evaluation consistency.
    # But we want different restarts. Let's use a counter.
    
    for k in range(1, 10):
        if time.time() > deadline - 10: # Leave 10s for final check
            break
        
        # Random initialization
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.full(N, 0.05)
        init_vars = np.column_stack((x, y, r)).flatten()
        
        # Quick local improvement to get a better start for SLSQP
        # This helps SLSQP converge faster
        for _ in range(100):
            for i in range(N):
                min_r = min(x[i], y[i], 1-x[i], 1-y[i])
                for j in range(N):
                    if i != j:
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        min_r = min(min_r, d - r[j])
                r[i] = max(0.01, min_r)
        
        init_vars = np.column_stack((x, y, r)).flatten()
        
        if time.time() > deadline - 5:
            break
            
        res = minimize(objective, init_vars, method='SLSQP', bounds=bounds, constraints=constraints_func, options={'maxiter': 300, 'ftol': 1e-8})
        if res.success and -res.fun > best_score:
            best_score = -res.fun
            best_vars = res.x

    if best_vars is None:
        # Fallback to grid if all failed
        return np.array([[(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3] for i in range(side) for j in range(side)][:N])
        
    # Extract results
    x = best_vars[0::3]
    y = best_vars[1::3]
    r = best_vars[2::3]
    
    # Final validation and clipping to ensure strict validity
    # Sometimes SLSQP might leave tiny violations
    for i in range(N):
        r[i] = max(0, r[i])
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
        
    # One last polish pass to fix any remaining overlaps due to numerical errors
    for _ in range(10):
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    # Push apart
                    if d == 0:
                        dx = 1.0
                        dy = 0.0
                        d = 1.0
                    else:
                        dx /= d
                        dy /= d
                    push = (min_d - d) / 2.0
                    x[i] += dx * push
                    y[i] += dy * push
                    x[j] -= dx * push
                    y[j] -= dy * push
                    # Clip to boundaries
                    x[i] = np.clip(x[i], r[i], 1 - r[i])
                    y[i] = np.clip(y[i], r[i], 1 - r[i])
                    x[j] = np.clip(x[j], r[j], 1 - r[j])
                    y[j] = np.clip(y[j], r[j], 1 - r[j])

    return np.column_stack((x, y, r))
