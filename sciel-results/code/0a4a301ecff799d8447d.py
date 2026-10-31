import numpy as np
import time

N = 21

def construct():
    # Simulated Annealing approach
    # Variables: x[0:N], y[0:N], W (H = 2 - W)
    # Objective: Maximize sum(r_i)
    
    def calc_radii_vectorized(x, y, W, H):
        # Wall distances
        r = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
        
        # Pairwise distances
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        
        # For each i, find min distance to any j != i
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        
        # r_i is min of wall dist and half min dist to neighbor
        r = np.minimum(r, min_dist / 2.0)
        
        # Ensure non-negative
        r = np.maximum(r, 0.0)
        
        return r

    def evaluate(x, y, W):
        H = 2.0 - W
        # Clamp positions to box
        x_clamped = np.clip(x, 0, W)
        y_clamped = np.clip(y, 0, H)
        r = calc_radii_vectorized(x_clamped, y_clamped, W, H)
        return np.sum(r), r

    def sa_optimize(x_init, y_init, W_init, max_iters=5000, T_start=0.1, T_end=0.001):
        x = x_init.copy()
        y = y_init.copy()
        W = W_init
        H = 2.0 - W
        
        # Initial evaluation
        score_curr, _ = evaluate(x, y, W)
        
        best_score = score_curr
        best_x = x.copy()
        best_y = y.copy()
        best_W = W
        
        # Temperature schedule
        for i in range(max_iters):
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Perturb a random circle's position
            k = np.random.randint(0, N)
            old_x = x[k]
            old_y = y[k]
            
            # Random step
            step = 0.1 * (1 - i/max_iters) + 0.01
            x[k] += np.random.uniform(-step, step)
            y[k] += np.random.uniform(-step, step)
            
            # Clamp
            x[k] = np.clip(x[k], 0, W)
            y[k] = np.clip(y[k], 0, H)
            
            new_score, _ = evaluate(x, y, W)
            
            delta = new_score - score_curr
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                score_curr = new_score
                if score_curr > best_score:
                    best_score = score_curr
                    best_x = x.copy()
                    best_y = y.copy()
                    best_W = W
            else:
                x[k] = old_x
                y[k] = old_y
                
        return best_score, best_x, best_y, best_W

    # Multi-start
    best_global_score = 0
    best_global_x = None
    best_global_y = None
    best_global_W = None
    
    # Try different aspect ratios (W)
    W_samples = np.linspace(0.6, 1.4, 9)
    
    # Time management
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    for W0 in W_samples:
        if time.time() - start_time > time_limit:
            break
            
        H0 = 2.0 - W0
        
        # Initialization: Random scatter within box
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            x_init = np.random.uniform(0, W0, N)
            y_init = np.random.uniform(0, H0, N)
            
            # Run SA
            score, x_opt, y_opt, W_opt = sa_optimize(x_init, y_init, W0, max_iters=5000)
            
            if score > best_global_score:
                best_global_score = score
                best_global_x = x_opt
                best_global_y = y_opt
                best_global_W = W_opt

    # If SA didn't find anything good (shouldn't happen), fallback
    if best_global_x is None:
        # Simple grid fallback
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(cols):
            for j in range(rows):
                if idx < N:
                    x[idx] = (i + 0.5) * W / cols
                    y[idx] = (j + 0.5) * H / rows
                    idx += 1
        r = calc_radii_vectorized(x, y, W, H)
        return np.column_stack((x, y, r))
        
    # Final calculation for the best result
    H_final = 2.0 - best_global_W
    r_final = calc_radii_vectorized(best_global_x, best_global_y, best_global_W, H_final)
    
    return np.column_stack((best_global_x, best_global_y, r_final))
