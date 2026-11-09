import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    # Precompute indices for vectorized overlap calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def validity_check(x, y, r):
        """Returns True if configuration is valid, False otherwise"""
        n = len(r)
        if np.any(r < -1e-12):
            return False
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return False
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        # Vectorized overlap check
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
            return False
        return True
    
    def make_valid(x, y, r):
        """Ensure configuration is valid by shrinking radii if needed"""
        r = r.copy()
        # First ensure bounds
        for i in range(N):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            if r[i] > max_r:
                r[i] = max(0, max_r)
        
        # Iteratively fix overlaps
        for _ in range(200):
            max_overlap = 0
            max_i, max_j = 0, 0
            # Vectorized overlap calculation for speed
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            overlaps = r[i_idx] + r[j_idx] - dist
            max_idx = np.argmax(overlaps)
            if overlaps[max_idx] <= 0:
                break
            max_overlap = overlaps[max_idx]
            # Get the actual indices
            max_i = i_idx[max_idx]
            max_j = j_idx[max_idx]
            
            # Shrink the larger circle by half the overlap plus a small buffer
            shrink = max_overlap / 2 + 1e-7
            if r[max_i] >= r[max_j]:
                r[max_i] = max(0, r[max_i] - shrink)
            else:
                r[max_j] = max(0, r[max_j] - shrink)
        
        # Final safety: if still invalid, scale down all radii
        if not validity_check(x, y, r):
            r *= 0.95
            for _ in range(50):
                if validity_check(x, y, r):
                    break
                r *= 0.9
        
        return x, y, r
    
    def maximize_radii(x, y, r, iterations=20):
        """Try to increase each radius to its maximum possible value"""
        eps = 1e-10
        r = r.copy()
        for _ in range(iterations):
            if time.time() > deadline:
                break
            changed = False
            for i in range(N):
                # Max radius from boundary
                r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
                # Max radius from neighbors
                r_neighbor = 0.5
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_neighbor = min(r_neighbor, max(0, dist - r[j] - eps))
                
                new_r = min(r_bound, r_neighbor)
                if new_r > r[i] + 1e-12:
                    r[i] = new_r
                    changed = True
            
            if not changed:
                break
        
        # Ensure validity after maximization
        if not validity_check(x, y, r):
            x, y, r = make_valid(x, y, r)
        
        return x, y, r
    
    def objective(params):
        """
        Objective function for L-BFGS-B.
        We want to maximize sum(r), so we minimize -sum(r).
        We add penalties for boundary violations and circle overlaps.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        # 1. Boundary Penalties
        pen_x_left = np.maximum(0, r - x)**2
        pen_x_right = np.maximum(0, x - (1 - r))**2
        pen_y_bottom = np.maximum(0, r - y)**2
        pen_y_top = np.maximum(0, y - (1 - r))**2
        boundary_pen = np.sum(pen_x_left + pen_x_right + pen_y_bottom + pen_y_top)
        
        # 2. Overlap Penalties (Vectorized)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d = np.sqrt(dx*dx + dy*dy)
        overlaps = r[i_idx] + r[j_idx] - d
        # Only penalize positive overlaps
        positive_overlaps = np.maximum(0, overlaps)
        overlap_pen = np.sum(positive_overlaps**2)
        
        # Weights: High weight to ensure validity is prioritized
        w_pen = 100.0
        
        return -np.sum(r) + w_pen * (boundary_pen + overlap_pen)
    
    def optimize_start(x0, y0, r0):
        """
        Run L-BFGS-B optimization from a given start.
        Returns the best valid configuration found.
        """
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Try to repair by making valid
            x_opt, y_opt, r_opt = make_valid(x_opt, y_opt, r_opt)
            
            # Try to maximize radii further
            x_opt, y_opt, r_opt = maximize_radii(x_opt, y_opt, r_opt)
            
            score = np.sum(r_opt)
            
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return -np.inf, None
    
    def local_search(x, y, r, num_iter=20):
        """
        Simple local search: perturb positions/radii, re-optimize.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for _ in range(num_iter):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            perturb_scale = 0.05
            x_new = x + np.random.normal(0, perturb_scale, N)
            y_new = y + np.random.normal(0, perturb_scale, N)
            r_new = r + np.random.normal(0, perturb_scale * 0.5, N)
            
            # Clip to bounds
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            r_new = np.clip(r_new, 0, 0.5)
            
            # Re-optimize from this new point
            score, config = optimize_start(x_new, y_new, r_new)
            
            if score > best_score:
                best_score = score
                if config is not None:
                    best_x = config[:, 0]
                    best_y = config[:, 1]
                    best_r = config[:, 2]
        
        return best_x, best_y, best_r, best_score
    
    # Initialization strategies
    def init_random():
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        return x, y, r
    
    def init_grid():
        # 5x5 grid is 25. Add 1 in center?
        x = []
        y = []
        r = []
        for i in range(5):
            y_pos = (i + 0.5) / 5
            for j in range(5):
                x_pos = (j + 0.5) / 5
                x.append(x_pos)
                y.append(y_pos)
                r.append(0.1)
        x.append(0.5)
        y.append(0.5)
        r.append(0.1)
        
        idx = np.random.permutation(N)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]
    
    def init_hex():
        # Hexagonal packing approximation
        x = []
        y = []
        r = []
        # Radius approx 0.1
        # Rows
        for i in range(6):
            y_pos = (i + 0.5) / 6
            if i % 2 == 0:
                # 5 circles
                for j in range(5):
                    x_pos = (j + 0.5) / 5
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.1)
            else:
                # 4 circles, shifted
                for j in range(4):
                    x_pos = (j + 1.0) / 5
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.1)
        # We have 5*3 + 4*3 = 15 + 12 = 27 circles. Need 26.
        # Let's just take the first 26.
        x = x[:26]
        y = y[:26]
        r = r[:26]
        idx = np.random.permutation(26)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]
    
    def polish(x, y, r):
        """
        High-precision polish pass.
        """
        params0 = np.concatenate([x, y, r])
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
        for i in range(N):
            bounds.append((0, 1))
        for i in range(N):
            bounds.append((0, 0.5))
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            x_opt, y_opt, r_opt = make_valid(x_opt, y_opt, r_opt)
            x_opt, y_opt, r_opt = maximize_radii(x_opt, y_opt, r_opt)
            
            return x_opt, y_opt, r_opt
        except Exception:
            return x, y, r

    # Main loop
    num_restarts = 1000 # We will break based on time
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        if restart % 3 == 1:
            x0, y0, r0 = init_random()
        elif restart % 3 == 2:
            x0, y0, r0 = init_grid()
        else:
            x0, y0, r0 = init_hex()
            
        # Run optimization
        score, config = optimize_start(x0, y0, r0)
        
        if score > best_score:
            best_score = score
            if config is not None:
                best_config = config.copy()
                
        # If we found a good score, do some local search to improve it
        if best_config is not None and score > 1.1 and time.time() < deadline - 30:
            x_ls, y_ls, r_ls, score_ls = local_search(best_config[:, 0], best_config[:, 1], best_config[:, 2], num_iter=10)
            if score_ls > best_score:
                best_score = score_ls
                best_config = np.column_stack((x_ls, y_ls, r_ls))

    # Final check
    if best_config is None:
        # Fallback
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        return np.column_stack((x, y, r))
    
    # Ensure final validity and polish
    x_f, y_f, r_f = best_config[:, 0], best_config[:, 1], best_config[:, 2]
    
    # Polish the best configuration
    x_f, y_f, r_f = polish(x_f, y_f, r_f)
    
    final_score = np.sum(r_f)
    
    if final_score < best_score:
        # If polish somehow reduced score (shouldn't happen if logic is correct, but safety first)
        # Re-check original
        if validity_check(best_config[:, 0], best_config[:, 1], best_config[:, 2]):
             final_score = best_score
             x_f, y_f, r_f = best_config[:, 0], best_config[:, 1], best_config[:, 2]
        else:
             # Last resort fallback
             x = np.random.uniform(0.1, 0.9, N)
             y = np.random.uniform(0.1, 0.9, N)
             r = np.full(N, 0.01)
             x_f, y_f, r_f = make_valid(x, y, r)
             x_f, y_f, r_f = maximize_radii(x_f, y_f, r_f)
    
    return np.column_stack((x_f, y_f, r_f))
