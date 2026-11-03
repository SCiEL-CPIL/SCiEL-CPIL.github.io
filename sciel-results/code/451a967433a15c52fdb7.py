import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Initial dimensions
    W = 1.0
    H = 1.0
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Phase 1: Sequential Insertion
    # Insert circles one by one, maximizing the radius of the new circle
    # subject to non-overlap with existing circles and boundary constraints.
    
    # Helper to calculate max radius for a circle at (cx, cy)
    def max_radius_at(cx, cy, W_curr, H_curr, x_prev, y_prev, r_prev, count):
        # Distance to walls
        r_wall = min(cx, W_curr - cx, cy, H_curr - cy)
        if r_wall <= 0:
            return 0.0
        
        # Distance to previous circles
        r_max = r_wall
        for i in range(count):
            dist = np.sqrt((cx - x_prev[i])**2 + (cy - y_prev[i])**2)
            max_r = dist - r_prev[i]
            if max_r < r_max:
                r_max = max_r
            if r_max <= 0:
                return 0.0
        return r_max

    # For each circle, find the best position
    for k in range(N):
        # Objective: maximize r, so minimize -r
        def neg_radius(pos):
            cx, cy = pos
            return -max_radius_at(cx, cy, W, H, x[:k], y[:k], r[:k], k)
        
        # Use multiple random starts for Nelder-Mead to find global max in this 2D space
        best_r = -np.inf
        best_pos = (0.5, 0.5)
        
        # Try a few random starting points
        np.random.seed(k) # Deterministic for reproducibility within run, but varies per k
        for _ in range(5):
            start = np.array([np.random.uniform(0.1, W-0.1), np.random.uniform(0.1, H-0.1)])
            res = minimize(neg_radius, start, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
            if res.success:
                r_val = -res.fun
                if r_val > best_r:
                    best_r = r_val
                    best_pos = res.x
        
        x[k] = best_pos[0]
        y[k] = best_pos[1]
        r[k] = best_r
        
    # Phase 2: Global Refinement
    # Optimize all x, y, r, W, H simultaneously
    # Variables: x (N), y (N), r (N), W, H
    x0 = np.concatenate([x, y, r, [W, H]])
    
    def objective(v):
        x_v = v[:N]
        y_v = v[N:2*N]
        r_v = v[2*N:3*N]
        W_v = v[3*N]
        H_v = v[3*N+1]
        
        # Soft penalty for invalid states to help optimizer navigate
        # But primarily rely on constraints. 
        # For SLSQP, we just return the objective.
        return -np.sum(r_v)
        
    def jac(v):
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    constraints = []
    
    # Perimeter constraint: 2*(W+H) = 4  => W+H = 2
    constraints.append({'type': 'eq', 'fun': lambda v: v[3*N] + v[3*N+1] - 2.0})
    
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })
            
    # Boundary constraints
    for i in range(N):
        # x_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        # x_i + r_i <= W
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
        # y_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        # y_i + r_i <= H
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
        
    # Radii > 0 (small epsilon to avoid singularities)
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i] - 1e-6})
        
    # W, H > 0
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N] - 0.1})
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1] - 0.1})

    # Bounds for variables to help SLSQP
    bounds = []
    for i in range(N):
        bounds.append((0, 2)) # x
    for i in range(N):
        bounds.append((0, 2)) # y
    for i in range(N):
        bounds.append((1e-6, 1.0)) # r
    bounds.append((0.5, 1.5)) # W
    bounds.append((0.5, 1.5)) # H

    res = minimize(objective, x0, jac=jac, method='SLSQP', constraints=constraints, bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
    
    if res.success:
        x_final = res.x[:N]
        y_final = res.x[N:2*N]
        r_final = res.x[2*N:3*N]
        W_final = res.x[3*N]
        H_final = res.x[3*N+1]
    else:
        # Fallback to sequential result if optimization fails
        x_final = x
        y_final = y
        r_final = r
        W_final = W
        H_final = H
        
    # Final validity check and scaling
    # Check if perimeter is violated due to floating point
    perim = 2 * (W_final + H_final)
    if abs(perim - 4) > 1e-5:
        # Scale to fit perimeter 4
        scale = 4.0 / perim
        x_final *= scale
        y_final *= scale
        r_final *= scale
        W_final *= scale
        H_final *= scale
        
    # Check for overlaps and boundary violations, shrink if necessary
    # Simple iterative shrinking if invalid
    for _ in range(10):
        valid = True
        for i in range(N):
            if x_final[i] < r_final[i] or x_final[i] + r_final[i] > W_final or y_final[i] < r_final[i] or y_final[i] + r_final[i] > H_final:
                valid = False
                break
            for j in range(i+1, N):
                dist_sq = (x_final[i]-x_final[j])**2 + (y_final[i]-y_final[j])**2
                rad_sum = r_final[i] + r_final[j]
                if dist_sq < rad_sum**2 - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        # Shrink all radii by 1%
        r_final *= 0.99
        
    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
