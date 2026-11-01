"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Objective function with penalty
    def objective(params):
        # params: (x_0, y_0, r_0, x_1, y_1, r_1, ..., x_31, y_31, r_31)
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Objective: maximize sum of radii -> minimize negative sum
        obj = -np.sum(r)
        
        # Penalty for circle-circle overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap * overlap
        
        # Penalty for boundary violations
        for i in range(N):
            # x boundary
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > 1.0 - r[i]:
                penalty += (x[i] - (1.0 - r[i]))**2
            # y boundary
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > 1.0 - r[i]:
                penalty += (y[i] - (1.0 - r[i]))**2
            # radius positivity
            if r[i] < 0:
                penalty += r[i]**2
        
        return obj + 100.0 * penalty

    def is_valid(x, y, r, tol=1e-6):
        # Check boundaries
        if np.any(x < r - tol) or np.any(x > 1.0 - r + tol) or np.any(y < r - tol) or np.any(y > 1.0 - r + tol) or np.any(r < 0):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def make_valid(x, y, r):
        # Simple projection: scale radii down if invalid
        # This is a heuristic fallback
        for _ in range(10):
            if is_valid(x, y, r):
                break
            # Find max violation
            max_viol = 0
            # Boundary violations
            for i in range(N):
                max_viol = max(max_viol, max(0, r[i] - x[i]), max(0, x[i] - (1-r[i])), max(0, r[i] - y[i]), max(0, y[i] - (1-r[i])))
            # Overlap violations
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    viol = r[i] + r[j] - dist
                    if viol > max_viol:
                        max_viol = viol
            
            if max_viol > 0:
                # Scale all radii down slightly
                scale = 1.0 - 0.01 * max_viol
                r = r * max(0.5, scale)
            else:
                break
        return x, y, r

    best_score = -np.inf
    best_result = None
    
    # Initial guess: 6x6 grid, take first 32
    side = 6
    xs_init = np.zeros(N)
    ys_init = np.zeros(N)
    rs_init = np.zeros(N)
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx < N:
                xs_init[idx] = (i + 0.5) / side
                ys_init[idx] = (j + 0.5) / side
                rs_init[idx] = 0.5 / side
                idx += 1
    
    x0 = np.concatenate([xs_init, ys_init, rs_init])
    
    # Run optimization
    # We might run multiple restarts if time allows
    num_restarts = 1
    # Check time
    if time.time() < deadline - 60:
        num_restarts = 3
    if time.time() < deadline - 120:
        num_restarts = 5
        
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            x0_current = x0.copy()
        else:
            # Perturb the best solution found so far or the initial grid
            if best_result is not None:
                x0_current = best_result.copy()
                x0_current += np.random.normal(0, 0.01, size=x0_current.shape)
            else:
                x0_current = x0 + np.random.normal(0, 0.01, size=x0.shape)
        
        # Box constraints for L-BFGS-B
        # x in [0,1], y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        res = minimize(objective, x0_current, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'maxfun': 5000})
        
        if res.success or res.fun < (best_score if best_result else np.inf):
            x_opt = res.x[0::3]
            y_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            # Post-process to ensure validity
            x_opt, y_opt, r_opt = make_valid(x_opt, y_opt, r_opt)
            
            if is_valid(x_opt, y_opt, r_opt):
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((x_opt, y_opt, r_opt))
    
    if best_result is None:
        # Fallback to initial grid
        best_result = np.column_stack((xs_init, ys_init, rs_init))
        
    return best_result
# EVOLVE-BLOCK-END
