import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_score = 0.0
    best_solution = None
    
    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r

    def pack(x, y, r):
        return np.concatenate([x, y, r])

    def is_valid(x, y, r, tol=1e-9):
        # Check boundaries
        if np.any(x < r - tol) or np.any(1 - x < r - tol):
            return False
        if np.any(y < r - tol) or np.any(1 - y < r - tol):
            return False
        if np.any(r < -tol):
            return False
            
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # dist >= r_sum for i != j
        np.fill_diagonal(dist, np.inf)
        if np.any(dist < r_sum - tol):
            return False
        return True

    def clearance_objective(v):
        """
        Objective: maximize sum of radii subject to constraints.
        We use a penalty method with smooth approximations.
        """
        x, y, r = unpack(v)
        
        # Sum of radii (to maximize, so we minimize negative)
        sum_r = np.sum(r)
        
        # Boundary violations: want x >= r, 1-x >= r, y >= r, 1-y >= r
        # Violation amount: max(0, r - x), max(0, r - (1-x)), etc.
        # Use smooth approximation: log(1 + exp(beta * violation)) / beta
        beta = 50.0
        
        def smooth_max(viol):
            return np.logaddexp(0, beta * viol) / beta
        
        pen_x = np.sum(smooth_max(r - x))
        pen_1mx = np.sum(smooth_max(r - (1 - x)))
        pen_y = np.sum(smooth_max(r - y))
        pen_1my = np.sum(smooth_max(r - (1 - y)))
        pen_bound = pen_x + pen_1mx + pen_y + pen_1my
        
        # Overlap violations: want dist(i,j) >= r(i) + r(j)
        # Violation: max(0, r(i)+r(j) - dist(i,j))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Only consider i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol = r_sum[mask] - dist[mask]
        pen_overlap = np.sum(smooth_max(viol))
        
        lam = 100.0
        total_penalty = lam * (pen_bound + pen_overlap)
        
        return -sum_r + total_penalty

    def get_initial_guesses():
        guesses = []
        
        # Hexagonal packing
        rows = 6
        cols = 6
        hex_x, hex_y, hex_r = [], [], []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5 + (i % 2) * 0.5) / cols
                y = (i + 0.5) / rows
                r = 0.5 / cols * 0.85
                if len(hex_x) < N:
                    hex_x.append(x)
                    hex_y.append(y)
                    hex_r.append(r)
        hex_x = np.array(hex_x[:N])
        hex_y = np.array(hex_y[:N])
        hex_r = np.array(hex_r[:N])
        guesses.append(pack(hex_x, hex_y, hex_r))
        
        # Square grid
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()[:N]
        grid_y = grid_y.flatten()[:N]
        grid_r = np.full(N, 0.5 / side * 0.85)
        guesses.append(pack(grid_x, grid_y, grid_r))
        
        # Random initializations
        for _ in range(4):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.08 + 0.02
            guesses.append(pack(x, y, r))
        
        return guesses

    guesses = get_initial_guesses()
    
    for v0 in guesses:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use Powell method which doesn't require gradients
            res = minimize(clearance_objective, v0, method='Powell', 
                          options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-10})
            
            if time.time() > deadline - 5:
                break
                
            v_opt = res.x
            x, y, r = unpack(v_opt)
            
            # Project to feasible region
            # Clip radii to be positive
            r = np.maximum(r, 1e-6)
            
            # For each circle, ensure it's within bounds
            # r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
            max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            r = np.minimum(r, max_r_bound)
            
            # Check validity and compute score
            if is_valid(x, y, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_solution = pack(x, y, r)
                    
        except Exception:
            pass
    
    # If we have time, do a polish pass with tighter tolerances
    if best_solution is not None and time.time() < deadline - 5:
        try:
            v_polish = best_solution.copy()
            res_polish = minimize(clearance_objective, v_polish, method='Powell',
                                 options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-12})
            
            if time.time() < deadline - 2:
                v_opt = res_polish.x
                x, y, r = unpack(v_opt)
                r = np.maximum(r, 1e-6)
                max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
                r = np.minimum(r, max_r_bound)
                
                if is_valid(x, y, r):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_solution = pack(x, y, r)
        except Exception:
            pass

    # If no valid solution found, return a trivial valid one
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r = unpack(best_solution)
    result = np.column_stack((x, y, r))
    return result
