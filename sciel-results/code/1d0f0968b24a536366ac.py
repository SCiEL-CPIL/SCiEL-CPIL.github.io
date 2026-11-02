"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # We'll optimize: for each circle i, variables are (x_i, y_i, r_i)
    # Plus rectangle dimensions: w, h with 2*(w+h) <= 4 => w+h <= 2
    # We can fix the rectangle to be as large as possible: w+h = 2
    # Let's parameterize w in [0.5, 1.5], h = 2 - w
    
    # Actually, let's think about this differently.
    # The optimal rectangle aspect ratio is unknown, so we should optimize it too.
    # But to keep it simpler, let's try a few aspect ratios and pick the best.
    
    # Variables: for N circles, each has (x, y, r) -> 3N variables
    # Plus 1 variable for aspect ratio (w, where h = 2 - w)
    # Total: 3N + 1 = 64 variables for N=21
    
    # Constraints:
    # 1. Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    # 2. Circle i fits in rectangle: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
    # 3. r_i > 0
    
    # Objective: maximize sum(r_i), i.e., minimize -sum(r_i)
    
    def objective(vars):
        # vars: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_N-1, y_N-1, r_N-1, w]
        rs = vars[2::3][:N]
        return -np.sum(rs)
    
    def constraints(vars):
        cons = []
        xs = vars[0::3][:N]
        ys = vars[1::3][:N]
        rs = vars[2::3][:N]
        w = vars[-1]
        h = 2.0 - w
        
        # Rectangle aspect ratio bounds
        cons.append({'type': 'ineq', 'fun': lambda v: v[-1] - 0.5})  # w >= 0.5
        cons.append({'type': 'ineq', 'fun': lambda v: 1.5 - v[-1]})  # w <= 1.5
        
        # Non-overlap constraints (soft for optimization)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
        
        # Circle fits in rectangle
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})  # x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[-1] - v[3*i] - v[3*i+2]})  # x_i <= w - r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})  # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[-1]) - v[3*i+1] - v[3*i+2]})  # y_i <= h - r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})  # r_i > 0
        
        return cons
    
    # This is too many constraints for SLSQP with N=21 (210 non-overlap + 84 bounds = ~294 constraints)
    # Let's use a penalty method instead
    
    def penalty_objective(vars, mu=10.0):
        xs = vars[0::3][:N]
        ys = vars[1::3][:N]
        rs = vars[2::3][:N]
        w = vars[-1]
        h = 2.0 - w
        
        penalty = 0.0
        
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist:
                    violation = min_dist - np.sqrt(dist_sq)
                    penalty += mu * violation * violation
        
        # Rectangle bounds
        for i in range(N):
            if xs[i] < rs[i]:
                penalty += mu * (rs[i] - xs[i])**2
            if xs[i] > w - rs[i]:
                penalty += mu * (xs[i] - (w - rs[i]))**2
            if ys[i] < rs[i]:
                penalty += mu * (rs[i] - ys[i])**2
            if ys[i] > h - rs[i]:
                penalty += mu * (ys[i] - (h - rs[i]))**2
            if rs[i] < 0:
                penalty += mu * rs[i]**2
        
        # w bounds
        if w < 0.5:
            penalty += mu * (0.5 - w)**2
        if w > 1.5:
            penalty += mu * (w - 1.5)**2
        
        return -np.sum(rs) + penalty
    
    # Initial guess: grid layout
    def initial_guess(w_init=1.0):
        h_init = 2.0 - w_init
        # Use a grid
        cols = int(np.ceil(np.sqrt(N * w_init / h_init)))
        rows = int(np.ceil(N / cols))
        init = []
        for i in range(N):
            col = i % cols
            row = i // cols
            x = (col + 0.5) * w_init / cols
            y = (row + 0.5) * h_init / rows
            r = min(w_init / cols, h_init / rows) / 2 * 0.99
            init.extend([x, y, r])
        init.append(w_init)
        return np.array(init)
    
    best_value = -np.inf
    best_vars = None
    
    # Try multiple initializations and aspect ratios
    for w_init in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        if time.time() > deadline:
            break
        
        x0 = initial_guess(w_init)
        
        try:
            result = minimize(penalty_objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-10})
            
            if result.success or result.fun < best_value:
                # Check validity
                xs = result.x[0::3][:N]
                ys = result.x[1::3][:N]
                rs = result.x[2::3][:N]
                w = result.x[-1]
                h = 2.0 - w
                
                valid = True
                if w < 0.5 - 1e-6 or w > 1.5 + 1e-6:
                    valid = False
                for i in range(N):
                    if rs[i] < -1e-6:
                        valid = False
                        break
                    if xs[i] < rs[i] - 1e-6 or xs[i] > w - rs[i] + 1e-6:
                        valid = False
                        break
                    if ys[i] < rs[i] - 1e-6 or ys[i] > h - rs[i] + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            if dx*dx + dy*dy < (rs[i] + rs[j])**2 - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    val = np.sum(rs)
                    if val > best_value:
                        best_value = val
                        best_vars = result.x.copy()
                        
        except Exception:
            pass
    
    if best_vars is not None:
        xs = best_vars[0::3][:N]
        ys = best_vars[1::3][:N]
        rs = best_vars[2::3][:N]
        result_array = np.column_stack([xs, ys, rs])
        return result_array
    else:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        cols = 5
        rows = 5
        result = []
        for i in range(N):
            col = i % cols
            row = i // cols
            x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            r = min(w / cols, h / rows) / 2 * 0.99
            result.append([x, y, r])
        return np.array(result)

# EVOLVE-BLOCK-END
