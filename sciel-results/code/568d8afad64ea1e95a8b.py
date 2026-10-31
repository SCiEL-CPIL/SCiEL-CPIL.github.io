"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _pack_with_admm_style(n, time_limit=140.0):
    """
    Pack n circles in a rectangle of perimeter 4 (w + h = 2) to maximize sum of radii.
    Uses a gradient-based approach with penalty for overlaps, alternating with constraint projection.
    """
    start_time = time.time()
    elapsed = 0.0
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(n)))
    # Rectangle dimensions: w=1, h=1 (perimeter 4)
    # But we can vary w and h as long as 2*(w+h)=4 => w+h=2
    # Let's start with w=1, h=1
    w, h = 1.0, 1.0
    
    # Place circles in a grid
    xs = []
    ys = []
    rs = []
    for i in range(side):
        for j in range(side):
            if len(xs) >= n:
                break
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            xs.append(x)
            ys.append(y)
            rs.append(r)
    
    xs = np.array(xs[:n])
    ys = np.array(ys[:n])
    rs = np.array(rs[:n])
    
    # Optimize using gradient descent with penalty
    # Variables: x, y, r for each circle, and w (h = 2 - w)
    # Constraint: w > 0, h > 0, circles inside rectangle, no overlaps
    
    best_sum = np.sum(rs)
    best_state = (xs.copy(), ys.copy(), rs.copy(), w)
    
    def objective_and_grad(vars_):
        # vars_: [x_1, y_1, r_1, ..., x_n, y_n, r_n, w]
        x = vars_[:n]
        y = vars_[n:2*n]
        r = vars_[2*n:3*n]
        w_val = vars_[3*n]
        h_val = 2.0 - w_val
        
        # Ensure positivity
        w_val = max(w_val, 0.1)
        h_val = max(h_val, 0.1)
        
        sum_r = np.sum(r)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        # Penalty for going outside
        for i in range(n):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] > w_val - r[i]:
                penalty += (x[i] - (w_val - r[i])) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] > h_val - r[i]:
                penalty += (y[i] - (h_val - r[i])) ** 2
        
        # Penalty for negative radius
        for i in range(n):
            if r[i] < 0:
                penalty += r[i] ** 2
        
        return -sum_r + 100.0 * penalty
    
    # Use Nelder-Mead for robustness
    from scipy.optimize import minimize
    
    x0 = np.concatenate([xs, ys, rs, [w]])
    
    result = minimize(objective_and_grad, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.fun < -best_sum + 100.0 * 0.1:  # Check if improved
        x_opt = result.x[:n]
        y_opt = result.x[n:2*n]
        r_opt = np.maximum(result.x[2*n:3*n], 0)
        w_opt = max(result.x[3*n], 0.1)
        
        # Verify constraints
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx = x_opt[i] - x_opt[j]
                dy = y_opt[i] - y_opt[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r_opt[i] + r_opt[j] - 1e-9:
                    valid = False
                    break
            if x_opt[i] < r_opt[i] - 1e-9 or x_opt[i] > w_opt - r_opt[i] + 1e-9:
                valid = False
                break
            if y_opt[i] < r_opt[i] - 1e-9 or y_opt[i] > (2.0 - w_opt) - r_opt[i] + 1e-9:
                valid = False
                break
        
        if valid:
            sum_r = np.sum(r_opt)
            if sum_r > best_sum:
                best_sum = sum_r
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)
    
    return best_state

def construct():
    start_time = time.time()
    n = N
    
    # Try multiple initial configurations
    best_state = None
    best_sum = -np.inf
    
    configs = []
    
    # Grid configurations with different aspect ratios
    for w_frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w = 2.0 * w_frac
        h = 2.0 - w
        side_x = max(1, int(np.sqrt(n * w / h)))
        side_y = max(1, int(np.sqrt(n * h / w)))
        
        xs = []
        ys = []
        rs = []
        for i in range(side_x):
            for j in range(side_y):
                if len(xs) >= n:
                    break
                x = (i + 0.5) * w / side_x
                y = (j + 0.5) * h / side_y
                r = 0.5 * min(w / side_x, h / side_y) * 0.95
                xs.append(x)
                ys.append(y)
                rs.append(r)
        
        if len(xs) >= n:
            xs = np.array(xs[:n])
            ys = np.array(ys[:n])
            rs = np.array(rs[:n])
            configs.append((xs, ys, rs, w))
    
    # Also try a simple 4x5 grid
    for aspect in [0.8, 1.0, 1.25]:
        w = 2.0 * aspect / (1 + aspect)
        h = 2.0 - w
        nx, ny = 4, 5
        xs = []
        ys = []
        rs = []
        for i in range(nx):
            for j in range(ny):
                if len(xs) >= n:
                    break
                x = (i + 0.5) * w / nx
                y = (j + 0.5) * h / ny
                r = 0.5 * min(w / nx, h / ny) * 0.95
                xs.append(x)
                ys.append(y)
                rs.append(r)
        
        if len(xs) >= n:
            xs = np.array(xs[:n])
            ys = np.array(ys[:n])
            rs = np.array(rs[:n])
            configs.append((xs, ys, rs, w))
    
    # Optimize each configuration
    for idx, (xs, ys, rs, w) in enumerate(configs):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        state = _pack_with_admm_style(n, time_limit=DEADLINE - SAFETY_MARGIN - (time.time() - start_time))
        if state is not None:
            x_opt, y_opt, r_opt, w_opt = state
            sum_r = np.sum(r_opt)
            if sum_r > best_sum:
                best_sum = sum_r
                best_state = state
    
    # Fallback: return a valid grid packing
    if best_state is None or best_sum <= 0:
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:n])
    
    x_opt, y_opt, r_opt, w_opt = best_state
    h_opt = 2.0 - w_opt
    
    # Build result array
    result = np.column_stack([x_opt, y_opt, r_opt])
    
    return result
# EVOLVE-BLOCK-END
