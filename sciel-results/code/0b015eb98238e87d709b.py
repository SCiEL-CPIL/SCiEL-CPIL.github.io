import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # We optimize x, y, r for N circles, and w (width). h = 2 - w.
    # Total variables: 3*N + 1
    
    def loss(params):
        # Unpack
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        
        # Ensure w and h are positive
        if w <= 0.1 or h <= 0.1:
            return 1e10
            
        loss_val = -np.sum(r)
        
        # Boundary penalties
        # x in [r, w-r], y in [r, h-r]
        # Violations:
        # x < r -> r - x
        # x > w - r -> x - (w - r)
        # y < r -> r - y
        # y > h - r -> y - (h - r)
        
        v_x_min = np.maximum(0, r - x)
        v_x_max = np.maximum(0, x - (w - r))
        v_y_min = np.maximum(0, r - y)
        v_y_max = np.maximum(0, y - (h - r))
        
        bound_pen = np.sum(v_x_min**2) + np.sum(v_x_max**2) + np.sum(v_y_min**2) + np.sum(v_y_max**2)
        
        # Overlap penalties
        # d_ij^2 >= (r_i + r_j)^2
        # Violation: (r_i + r_j)^2 - d_ij^2 if positive
        # Using squared distance difference for smoothness?
        # Let's use max(0, (r_i+r_j) - d_ij)^2
        
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap_pen += (min_d - d)**2
        
        # Weights
        w_bound = 10.0
        w_overlap = 10.0
        
        return loss_val + w_bound * bound_pen + w_overlap * overlap_pen

    def check_validity(params, tol=1e-5):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
            
        # Check boundaries
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                min_d2 = (r[i] + r[j])**2
                if d2 < min_d2 - tol:
                    return False, 0.0
                    
        return True, np.sum(r)

    # Initial guess: Grid
    # Try a few aspect ratios or just optimize w
    # Let's start with w=1.0 (square)
    
    best_score = -1
    best_params = None
    
    # Number of restarts
    # Each optimization might take 1-5 seconds.
    # 140s / 3s ~ 40 restarts.
    max_restarts = 30
    
    # Pre-compute grid positions for different w
    # We will jitter the grid.
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Randomize aspect ratio slightly or use a grid
        # Let's use a grid of 5x5 (25 cells) and pick 21, or 4x6 (24) pick 21?
        # 21 is 3x7 or similar.
        # Let's just place them in a grid within the current w, h.
        # For simplicity, start with w=1.0, h=1.0.
        
        w_init = 1.0 + 0.1 * (np.random.rand() - 0.5) # Small variation
        h_init = 2.0 - w_init
        
        # Grid placement
        # 5 columns, 5 rows -> 25 spots. Pick 21.
        # Or 3 columns, 7 rows.
        # Let's use a dense grid and jitter.
        cols = 5
        rows = 5
        x_init = np.zeros(N)
        y_init = np.zeros(N)
        r_init = np.zeros(N)
        
        # Fill grid
        idx = 0
        for i in range(cols):
            for j in range(rows):
                if idx >= N:
                    break
                x_init[idx] = (i + 0.5) * (w_init / cols)
                y_init[idx] = (j + 0.5) * (h_init / rows)
                # Initial radius estimate: min(w/cols, h/rows) / 2
                r_init[idx] = 0.5 * min(w_init / cols, h_init / rows) * 0.9
                idx += 1
                
        # Add jitter
        x_init += np.random.rand(N) * 0.05 * (w_init / cols)
        y_init += np.random.rand(N) * 0.05 * (h_init / rows)
        
        # Initial params
        params0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        
        # Optimize
        try:
            res = minimize(loss, params0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-6})
            # Check validity
            valid, score = check_validity(res.x)
            if valid and score > best_score:
                best_score = score
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to simple grid
        w = 1.0
        h = 1.0
        cols = 5
        rows = 5
        cells = []
        for i in range(cols):
            for j in range(rows):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * w / cols
                y = (j + 0.5) * h / rows
                r = 0.5 * min(w / cols, h / rows) * 0.9
                cells.append([x, y, r])
        return np.array(cells[:N])
        
    # Extract result
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
