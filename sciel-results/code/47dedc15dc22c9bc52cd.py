import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 150.0  # Seconds - use full budget with safety margin

def construct():
    start_time = time.time()
    best_score = 0.0
    best_params = None
    
    # Lambda for penalty
    lam = 500.0
    # Epsilon for smooth max
    eps = 1e-4
    
    def smooth_max(u):
        return (u + np.sqrt(u**2 + eps**2)) / 2.0

    def objective(params):
        w = params[0]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10
        
        x = params[1:N+1]
        y = params[N+1:2*N+1]
        r = params[2*N+1:3*N+1]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Box violations
        box_viol = 0.0
        for i in range(N):
            # x bounds
            box_viol += smooth_max(r[i] - x[i])**2
            box_viol += smooth_max(w - r[i] - x[i])**2
            # y bounds
            box_viol += smooth_max(r[i] - y[i])**2
            box_viol += smooth_max(h - r[i] - y[i])**2
            
        # Overlap violations
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                viol = r[i] + r[j] - dist
                overlap_viol += smooth_max(viol)**2
                
        penalty = lam * (box_viol + overlap_viol)
        return -sum_r + penalty

    # Bounds
    bounds = [(0.2, 1.8)] # w
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
    for i in range(N):
        bounds.append((0.0, 2.0)) # y
    for i in range(N):
        bounds.append((0.0, 1.0)) # r

    def check_validity(x, y, r, w, h, tol=1e-6):
        # Check box
        for i in range(N):
            if r[i] > x[i] + tol or r[i] > w - x[i] + tol or \
               r[i] > y[i] + tol or r[i] > h - y[i] + tol:
                return False
        # Check overlap
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if r[i] + r[j] > dist + tol:
                    return False
        return True

    def polish(params, max_iter=2000):
        """Final high-precision polish pass with tighter tolerance"""
        try:
            res = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            return res.x
        except Exception:
            return params

    # We will try a few different aspect ratios and grid layouts
    num_restarts = 50
    restarts_done = 0
    best_polished = None
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
            
        # Vary initial aspect ratio more aggressively
        # Try a range of aspect ratios from 0.5 to 2.0
        if restart < 15:
            w0 = 0.5 + (restart / 15.0) * 1.5  # 0.5 to 2.0
        elif restart < 30:
            # Focus on promising ranges
            w0 = 0.6 + 0.8 * np.random.rand()  # 0.6 to 1.4
        else:
            # Random aspect ratios
            w0 = 0.4 + 1.6 * np.random.rand()
        
        # Initialize grid
        # Use a grid that adapts to aspect ratio
        # Number of columns and rows based on aspect ratio
        aspect = w0 / (2.0 - w0)
        if aspect > 1:
            cols = int(np.ceil(np.sqrt(N * aspect)))
        else:
            cols = int(np.ceil(np.sqrt(N / aspect)))
        cols = max(1, min(cols, N))
        rows = int(np.ceil(N / cols))
        
        h0 = 2.0 - w0
        cell_w = w0 / cols
        cell_h = h0 / rows
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x0[i] = (c + 0.5) * cell_w
            y0[i] = (r_idx + 0.5) * cell_h
            r0[i] = 0.45 * min(cell_w, cell_h)
        
        # Add small noise
        x0 += np.random.normal(0, 0.02, N)
        y0 += np.random.normal(0, 0.02, N)
        r0 += np.random.normal(0, 0.005, N)
        
        # Clip to valid ranges
        x0 = np.clip(x0, 0, w0)
        y0 = np.clip(y0, 0, h0)
        r0 = np.clip(r0, 0.01, 0.5)
        
        params0 = np.concatenate(([w0], x0, y0, r0))
        
        # Optimize
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
            params_opt = res.x
            w_opt = params_opt[0]
            h_opt = 2.0 - w_opt
            x_opt = params_opt[1:N+1]
            y_opt = params_opt[N+1:2*N+1]
            r_opt = params_opt[2*N+1:3*N+1]
            
            # Check validity
            valid = check_validity(x_opt, y_opt, r_opt, w_opt, h_opt)
            
            if valid:
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_params = (x_opt.copy(), y_opt.copy(), r_opt.copy())
                    # Store for potential polish
                    best_polished = params_opt.copy()
                    
        except Exception as e:
            pass
        restarts_done += 1
    
    # Final polish pass on the best solution if we have time
    if best_polished is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            try:
                polished_params = polish(best_polished, max_iter=3000)
                w_p = polished_params[0]
                h_p = 2.0 - w_p
                x_p = polished_params[1:N+1]
                y_p = polished_params[N+1:2*N+1]
                r_p = polished_params[2*N+1:3*N+1]
                
                if check_validity(x_p, y_p, r_p, w_p, h_p):
                    score_p = np.sum(r_p)
                    if score_p > best_score:
                        best_score = score_p
                        best_params = (x_p.copy(), y_p.copy(), r_p.copy())
            except Exception:
                pass

    if best_params is None:
        # Fallback: return a simple valid grid
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                x0 = cx * w
                y0 = cy * h
                r0 = 0.5 / side - 1e-3
                cells.append([x0, y0, r0])
        np.random.shuffle(cells)
        x = np.array([c[0] for c in cells[:N]])
        y = np.array([c[1] for c in cells[:N]])
        r = np.array([c[2] for c in cells[:N]])
        return np.column_stack((x, y, r))
        
    x, y, r = best_params
    return np.column_stack((x, y, r))
