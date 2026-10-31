import numpy as np
import time

N = 26

def construct():
    """
    Packs 26 circles in a unit square maximizing the sum of radii.
    Uses Simulated Annealing with multiple restarts.
    """
    best_valid_sum = -1
    best_valid_x = None
    best_valid_y = None
    best_valid_r = None
    
    # Parameters
    num_restarts = 5
    max_time_per_restart = 20.0
    
    for restart in range(num_restarts):
        # Initialization
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.02)
        
        def calc_cost(x, y, r):
            obj = -np.sum(r)
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            violations = np.maximum(0, r_sum - dist)
            np.fill_diagonal(violations, 0)
            penalty_overlap = np.sum(violations**2)
            
            pen_x_low = np.maximum(0, r - x)
            pen_x_high = np.maximum(0, x - (1 - r))
            pen_y_low = np.maximum(0, r - y)
            pen_y_high = np.maximum(0, y - (1 - r))
            penalty_boundary = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                              np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
            
            return obj + 1000.0 * (penalty_overlap + penalty_boundary)
        
        current_cost = calc_cost(x, y, r)
        
        T = 1.0
        alpha = 0.9995
        step_size = 0.05
        iterations = 100000
        start_time = time.time()
        
        for i in range(iterations):
            if time.time() - start_time > max_time_per_restart:
                break
            
            idx = np.random.randint(N)
            
            if np.random.random() < 0.7:
                dx_move = np.random.normal(0, step_size)
                dy_move = np.random.normal(0, step_size)
                new_x = x.copy()
                new_y = y.copy()
                new_r = r.copy()
                new_x[idx] = np.clip(x[idx] + dx_move, 0, 1)
                new_y[idx] = np.clip(y[idx] + dy_move, 0, 1)
            else:
                dr_move = np.random.normal(0, step_size * 0.5)
                new_x = x.copy()
                new_y = y.copy()
                new_r = r.copy()
                new_r[idx] = np.clip(r[idx] + dr_move, 0, 0.5)
            
            new_cost = calc_cost(new_x, new_y, new_r)
            
            delta = new_cost - current_cost
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x = new_x
                y = new_y
                r = new_r
                current_cost = new_cost
                
                # Check validity periodically
                if i % 500 == 0:
                    # Vectorized validity check
                    dx = x[:, np.newaxis] - x[np.newaxis, :]
                    dy = y[:, np.newaxis] - y[np.newaxis, :]
                    dist = np.sqrt(dx**2 + dy**2)
                    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
                    np.fill_diagonal(r_sum, 0)
                    
                    # Overlap check
                    if not np.any(dist < r_sum - 1e-8):
                        # Boundary check
                        if not (np.any(x < r - 1e-8) or np.any(x > 1 - r + 1e-8) or 
                                np.any(y < r - 1e-8) or np.any(y > 1 - r + 1e-8)):
                            s = np.sum(r)
                            if s > best_valid_sum:
                                best_valid_sum = s
                                best_valid_x = x.copy()
                                best_valid_y = y.copy()
                                best_valid_r = r.copy()
            
            T *= alpha
            step_size *= 0.9999
            
    if best_valid_x is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.column_stack((best_valid_x, best_valid_y, best_valid_r))
