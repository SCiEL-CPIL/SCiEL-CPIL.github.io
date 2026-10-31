import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_score = 0.0
    best_params = None
    
    # Try different aspect ratios for the rectangle w x h
    # w + h = 2
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8] # 0.8 is 1/1.25
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        if ar >= 1:
            w = 2 * ar / (1 + ar)
            h = 2 / (1 + ar)
        else:
            w = 2 / (1 + 1/ar) # This logic is simpler: w = 2*ar/(1+ar) works for all if ar = w/h
            # Let's just use w = 2 * ar / (1 + ar) and h = 2 / (1 + ar)
            # If ar < 1, w < h.
            w = 2 * ar / (1 + ar)
            h = 2 / (1 + ar)
            
        # Initial guess: Grid layout
        # Estimate number of rows and cols
        # Try to fit N circles in a grid
        # Approximate radius if packed in grid
        # Let's just initialize with small radii and random positions in the box
        # Or better: initialize on a grid of cells
        
        # Determine grid size
        # We want roughly square cells.
        # Let's try a few grid configurations for initialization
        init_attempts = 0
        while init_attempts < 3 and time.time() < deadline:
            init_attempts += 1
            # Randomize the grid slightly or use different grid shapes
            # Simple grid: cols = ceil(sqrt(N * w/h))
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            
            # Adjust cols/rows to be close to N
            # Just use a fixed grid that is at least N
            # If cols*rows < N, increase one
            while cols * rows < N:
                if w > h:
                    cols += 1
                else:
                    rows += 1
            
            # Positions
            x = np.linspace(w / (2*cols), w - w / (2*cols), cols)
            y = np.linspace(h / (2*rows), h - h / (2*rows), rows)
            
            xx, yy = np.meshgrid(x, y)
            xx = xx.flatten()[:N]
            yy = yy.flatten()[:N]
            
            # Initial radii: small fraction of cell size
            cell_w = w / cols
            cell_h = h / rows
            r_init = 0.4 * min(cell_w, cell_h)
            r = np.full(N, r_init)
            
            params = np.concatenate([xx, yy, r])
            
            # Objective function
            def objective(p):
                x_opt = p[0:N]
                y_opt = p[N:2*N]
                r_opt = p[2*N:3*N]
                
                # Boundary penalties
                # x must be in [r, w-r]
                # y must be in [r, h-r]
                pen_x = np.sum(np.maximum(0, r_opt - x_opt)**2) + np.sum(np.maximum(0, r_opt - (w - x_opt))**2)
                pen_y = np.sum(np.maximum(0, r_opt - y_opt)**2) + np.sum(np.maximum(0, r_opt - (h - y_opt))**2)
                pen_boundary = pen_x + pen_y
                
                # Overlap penalties
                # dist^2 >= (r_i + r_j)^2
                pen_overlap = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_opt[i] - x_opt[j]
                        dy = y_opt[i] - y_opt[j]
                        dist_sq = dx*dx + dy*dy
                        sum_r = r_opt[i] + r_opt[j]
                        if dist_sq < sum_r*sum_r:
                            # Penalty proportional to overlap depth squared
                            # Approximate overlap depth as (sum_r - dist)
                            dist = np.sqrt(dist_sq)
                            overlap = sum_r - dist
                            pen_overlap += overlap*overlap
                
                # We want to maximize sum(r), so minimize -sum(r)
                # Plus large penalty for violations
                val = -np.sum(r_opt) + 10.0 * (pen_boundary + pen_overlap)
                return val
            
            # Run optimization
            try:
                res = minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 500, 'maxfun': 5000})
            except Exception:
                continue
                
            if time.time() > deadline:
                break
                
            if res.success or res.fun < 1e-3:
                x_opt = res.x[0:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Verify validity
                valid = True
                # Check boundaries
                if np.any(r_opt > x_opt) or np.any(r_opt > y_opt) or \
                   np.any(r_opt > w - x_opt) or np.any(r_opt > h - y_opt):
                    # Allow tiny numerical errors
                    if np.any(r_opt > x_opt + 1e-5) or np.any(r_opt > y_opt + 1e-5) or \
                       np.any(r_opt > w - x_opt + 1e-5) or np.any(r_opt > h - y_opt + 1e-5):
                        valid = False
                
                if valid:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x_opt[i] - x_opt[j]
                            dy = y_opt[i] - y_opt[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r_opt[i] + r_opt[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        score = np.sum(r_opt)
                        if score > best_score:
                            best_score = score
                            best_params = np.column_stack((x_opt, y_opt, r_opt))
                            
    if best_params is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    else:
        return best_params
