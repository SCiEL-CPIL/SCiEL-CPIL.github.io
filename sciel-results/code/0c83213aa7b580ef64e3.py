import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    # Initial layout: 3-3-4 grid
    centers = np.array([
        [1/6, 1/6], [1/2, 1/6], [5/6, 1/6],
        [1/6, 1/2], [1/2, 1/2], [5/6, 1/2],
        [0.125, 19/24], [0.375, 19/24], [0.625, 19/24], [0.875, 19/24]
    ], dtype=np.float64)
    sides = np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    
    # Pack into vector: [cx0, cy0, s0, cx1, cy1, s1, ...]
    def unpack(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        return c, s
    
    def overlap_penalty(c, s):
        """Compute penalty for overlaps and boundary violations."""
        pen = 0.0
        # Boundary constraints
        for i in range(n):
            if c[i, 0] - s[i]/2 < -tol:
                pen += (c[i, 0] - s[i]/2 + tol) ** 2
            if c[i, 0] + s[i]/2 > 1 + tol:
                pen += (c[i, 0] + s[i]/2 - 1 - tol) ** 2
            if c[i, 1] - s[i]/2 < -tol:
                pen += (c[i, 1] - s[i]/2 + tol) ** 2
            if c[i, 1] + s[i]/2 > 1 + tol:
                pen += (c[i, 1] + s[i]/2 - 1 - tol) ** 2
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                # If they overlap (|dx| < t and |dy| < t), add penalty
                if abs(dx) < t and abs(dy) < t:
                    # Penalty for being too close
                    pen += (t - abs(dx)) ** 2 + (t - abs(dy)) ** 2
        
        return pen
    
    def objective(x):
        c, s = unpack(x)
        # Maximize sum of sides, minimize penalty
        return -np.sum(s) + 100 * overlap_penalty(c, s)
    
    def constraint_boundary(x):
        c, s = unpack(x)
        # All constraints >= 0
        cons = []
        for i in range(n):
            cons.append(c[i, 0] - s[i]/2 + tol)
            cons.append(1 + tol - c[i, 0] - s[i]/2)
            cons.append(c[i, 1] - s[i]/2 + tol)
            cons.append(1 + tol - c[i, 1] - s[i]/2)
            cons.append(s[i])
        return cons
    
    # Initial packed vector
    x0 = np.concatenate([centers.flatten(), sides])
    
    # Bounds: centers in [-0.1, 1.1], sides in [0.01, 1.0]
    bounds = []
    for i in range(n):
        bounds.append((-0.1, 1.1))  # cx
        bounds.append((-0.1, 1.1))  # cy
    for i in range(n):
        bounds.append((0.01, 1.0))  # s
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try multiple optimization strategies
    strategies = ['L-BFGS-B', 'SLSQP', 'Powell']
    
    for strat in strategies:
        if time.time() > deadline:
            break
        
        try:
            if strat == 'SLSQP':
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=[{'type': 'ineq', 'fun': constraint_boundary}],
                              options={'maxiter': 200, 'ftol': 1e-10})
            else:
                res = minimize(objective, x0, method=strat, bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Polish with high-precision local search
    if time.time() < deadline:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Extract final solution
    c_final, s_final = unpack(best_x)
    
    # Clamp to valid range
    for i in range(n):
        s_final[i] = max(0.01, min(1.0, s_final[i]))
        c_final[i, 0] = max(s_final[i]/2 - tol, min(1 - s_final[i]/2 + tol, c_final[i, 0]))
        c_final[i, 1] = max(s_final[i]/2 - tol, min(1 - s_final[i]/2 + tol, c_final[i, 1]))
    
    # Final validation and repair
    for i in range(n):
        for j in range(i+1, n):
            t = (s_final[i] + s_final[j]) / 2 - tol
            dx = c_final[i, 0] - c_final[j, 0]
            dy = c_final[i, 1] - c_final[j, 1]
            if abs(dx) < t and abs(dy) < t:
                # Nudge apart
                if abs(dx) < abs(dy):
                    if dx >= 0:
                        c_final[i, 0] += (t - abs(dx)) / 2 + 1e-9
                        c_final[j, 0] -= (t - abs(dx)) / 2 + 1e-9
                    else:
                        c_final[i, 0] -= (t - abs(dx)) / 2 + 1e-9
                        c_final[j, 0] += (t - abs(dx)) / 2 + 1e-9
                else:
                    if dy >= 0:
                        c_final[i, 1] += (t - abs(dy)) / 2 + 1e-9
                        c_final[j, 1] -= (t - abs(dy)) / 2 + 1e-9
                    else:
                        c_final[i, 1] -= (t - abs(dy)) / 2 + 1e-9
                        c_final[j, 1] += (t - abs(dy)) / 2 + 1e-9
    
    # Re-clamp after repair
    for i in range(n):
        c_final[i, 0] = max(s_final[i]/2 - tol, min(1 - s_final[i]/2 + tol, c_final[i, 0]))
        c_final[i, 1] = max(s_final[i]/2 - tol, min(1 - s_final[i]/2 + tol, c_final[i, 1]))
    
    # If result is worse than initial, fall back to initial
    initial_sum = np.sum(sides)
    final_sum = np.sum(s_final)
    if final_sum < initial_sum - 1e-6:
        c_final = centers.copy()
        s_final = sides.copy()
    
    return c_final, s_final
