import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_val = -np.inf
    best_x = None
    
    # Objective function
    def objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Penalty for boundary violations
        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0, r[i] - x[i])**2
            pen_bound += max(0, r[i] - y[i])**2
            pen_bound += max(0, r[i] - (1 - x[i]))**2
            pen_bound += max(0, r[i] - (1 - y[i]))**2
            
        # Penalty for overlaps
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    pen_overlap += overlap**2
                    
        # Maximize sum of radii -> Minimize negative sum
        val = -np.sum(r) + 100.0 * (pen_bound + pen_overlap)
        return val

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99)) # x
    for i in range(N):
        bounds.append((0.01, 0.99)) # y
    for i in range(N):
        bounds.append((0.01, 0.5))  # r

    # Multi-start
    np.random.seed(42)
    n_starts = 100
    
    for k in range(n_starts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        # Mix of random and structured starts
        if k % 2 == 0:
            # Random
            x0 = np.random.uniform(0.1, 0.9, N)
            y0 = np.random.uniform(0.1, 0.9, N)
            r0 = np.random.uniform(0.05, 0.15, N)
        else:
            # Grid-like with noise
            side = int(np.ceil(np.sqrt(N)))
            xs, ys = [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
            x0 = np.array(xs) + np.random.uniform(-0.05, 0.05, N)
            y0 = np.array(ys) + np.random.uniform(-0.05, 0.05, N)
            r0 = np.random.uniform(0.1, 0.2, N)
            
        params0 = np.concatenate([x0, y0, r0])
        
        res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        
        if res.success:
            x_opt = res.x[0:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Check validity
            valid = True
            for i in range(N):
                if r_opt[i] > x_opt[i] + 1e-5 or r_opt[i] > y_opt[i] + 1e-5 or r_opt[i] > 1 - x_opt[i] + 1e-5 or r_opt[i] > 1 - y_opt[i] + 1e-5:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                        if r_opt[i] + r_opt[j] > dist + 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()

    if best_x is None:
        # Fallback to seed if no valid solution found (should not happen)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Post-processing: Ensure strict validity by slightly shrinking if needed
    x = best_x[0:N]
    y = best_x[N:2*N]
    r = best_x[2*N:3*N]
    
    # Shrink radii slightly to ensure strict inequality for disjoint interiors
    # The problem asks for disjoint interiors, so touching is allowed?
    # "disjoint interiors" means they can touch.
    # But numerical errors might cause slight overlap.
    # Let's check and shrink if overlap exists.
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            sum_r = r[i] + r[j]
            if sum_r > dist:
                # Scale down both radii slightly
                # Simple fix: reduce the larger one or both
                factor = dist / sum_r
                r[i] *= factor
                r[j] *= factor
        # Also check bounds
        r[i] = min(r[i], x[i], y[i], 1-x[i], 1-y[i])
        
    # Final check and return
    result = np.column_stack((x, y, r))
    return result
# EVOLVE-BLOCK-END
