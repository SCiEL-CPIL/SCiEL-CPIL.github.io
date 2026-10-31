import numpy as np
from scipy.optimize import minimize
import time

N = 26

def _validity_check(circles):
    """
    Check if the configuration of circles is valid.
    circles: array of shape (N, 3) with (x, y, r)
    Returns: True if valid, False otherwise.
    """
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        # Check bounds
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
        if r < -1e-9:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            radius_sum = ri + rj
            if dist_sq < radius_sum**2 - 1e-9:
                return False
    return True

def _score(circles):
    return np.sum(circles[:, 2])

def _objective(vars):
    """
    Objective function for minimization.
    We want to maximize sum of radii, so minimize -sum(r).
    """
    n = N
    x = vars[0:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    return -np.sum(r)

def _constraints(vars):
    """
    List of constraint functions for SLSQP.
    Each constraint is a dict: {'type': 'ineq', 'fun': ...}
    fun must return >= 0 for feasibility.
    """
    n = N
    x = vars[0:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    cons = []
    
    # Bounds: x - r >= 0, 1 - x - r >= 0, etc.
    # We can handle simple bounds with bounds argument in minimize, 
    # but r is coupled with x,y. 
    # Let's use inequality constraints for all.
    
    for i in range(n):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*n+i]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[n+i] - v[2*n+i]})
        # r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*n+i]})
        
    # Overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i] - v[j])**2 + (v[n+i] - v[n+j])**2 - (v[2*n+i] + v[2*n+j])**2
            })
            
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_circles = None
    best_score = -np.inf
    
    # Initial configurations
    # 1. Grid 5x5 + 1
    # 2. Grid 6x6 (first 26)
    # 3. Random
    
    def make_grid_config(rows, cols, n_circles):
        x_coords = []
        y_coords = []
        for i in range(rows):
            for j in range(cols):
                x_coords.append((j + 0.5) / cols)
                y_coords.append((i + 0.5) / rows)
        
        # Shuffle to pick random n_circles if needed, or just take first
        indices = np.arange(rows * cols)
        np.random.shuffle(indices)
        selected = indices[:n_circles]
        
        x = np.array([x_coords[k] for k in selected])
        y = np.array([y_coords[k] for k in selected])
        
        # Initial radius: conservative
        # Min distance in grid is 1/max(rows, cols)
        # So radius can be approx 0.5 / max(rows, cols)
        r = np.full(n_circles, 0.5 / max(rows, cols) * 0.8)
        
        circles = np.column_stack((x, y, r))
        return circles

    def make_random_config():
        # Greedy random placement
        circles = []
        np.random.seed(42) # Fixed seed for reproducibility of the random restart itself? 
                           # No, we want different restarts. Let's use global random state.
        
        # Try to place 26 circles
        # Start with large radius, try to place, if fail, retry with smaller?
        # This is complex. Let's just use a jittered grid for randomization.
        return None

    # Strategy: Multiple restarts with jittered grids
    # We will try 5x5 grid and 6x6 grid with random jitter
    
    attempts = 0
    max_attempts = 50 # Cap attempts, but time is the real limit
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Choose a base grid
        if attempts % 2 == 1:
            # 5x5 grid (25 circles) + 1 random
            base_r = 0.5 / 5
            x = np.array([(j + 0.5) / 5 for j in range(5) for i in range(5)])
            y = np.array([(i + 0.5) / 5 for j in range(5) for i in range(5)])
            r = np.full(25, base_r * 0.9)
            
            # Add 26th circle
            x_new = np.random.rand()
            y_new = np.random.rand()
            r_new = base_r * 0.5
            x = np.append(x, x_new)
            y = np.append(y, y_new)
            r = np.append(r, r_new)
        else:
            # 6x6 grid, pick 26
            base_r = 0.5 / 6
            x_all = np.array([(j + 0.5) / 6 for j in range(6) for i in range(6)])
            y_all = np.array([(i + 0.5) / 6 for j in range(6) for i in range(6)])
            indices = np.arange(36)
            np.random.shuffle(indices)
            selected = indices[:26]
            x = x_all[selected]
            y = y_all[selected]
            r = np.full(26, base_r * 0.9)
            
        # Add jitter
        jitter = 0.05
        x = x + np.random.uniform(-jitter, jitter, N)
        y = y + np.random.uniform(-jitter, jitter, N)
        
        # Clip to valid range for initial guess
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        vars0 = np.concatenate((x, y, r))
        
        # Run optimization
        try:
            res = minimize(
                _objective, 
                vars0, 
                method='SLSQP', 
                bounds=[(0, 1)] * (2 * N) + [(0, 0.5)] * N, # x,y in [0,1], r in [0,0.5]
                constraints=_constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success:
                vars_opt = res.x
                x_opt = vars_opt[0:N]
                y_opt = vars_opt[N:2*N]
                r_opt = vars_opt[2*N:3*N]
                circles_opt = np.column_stack((x_opt, y_opt, r_opt))
                
                # Check validity
                if _validity_check(circles_opt):
                    score = _score(circles_opt)
                    if score > best_score:
                        best_score = score
                        best_circles = circles_opt
        except Exception as e:
            pass
            
    if best_circles is None:
        # Fallback to simple grid if optimization failed completely
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        # Add one more
        cells.append((0.5, 0.5, 0.1)) # Center, might overlap, need to be careful
        # Just return the 25 grid + a tiny one in a corner?
        # Better: return the 5x5 grid and one tiny circle at (0.1, 0.1)
        # Actually, just return the first 26 from a 6x6 grid as fallback
        side = 6
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles
