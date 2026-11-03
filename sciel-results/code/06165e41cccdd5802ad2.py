import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _max_radius_for_circle(i, pos, radii, w, h):
    """Compute the maximum radius circle i can have given other circles' positions/radii."""
    # Wall constraints
    max_r = min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
    if max_r <= 0:
        return 0.0
    # Circle-circle constraints: r_i <= d - r_j for each j
    for j in range(N):
        if j == i:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.sqrt(dx * dx + dy * dy)
        r_max = d - radii[j]
        if r_max < max_r:
            max_r = r_max
    return max(0.0, max_r)

def _refine_positions(pos, radii, w, h, max_iters=200, tol=1e-9):
    """
    Uses SLSQP to jointly optimize positions and radii to maximize sum of radii.
    Variables: x = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = N
    
    # Initial guess
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pos[i, 0]
        x0[3*i+1] = pos[i, 1]
        x0[3*i+2] = radii[i]
        
    # Objective: maximize sum of radii -> minimize -sum(radii)
    def objective(x):
        return -np.sum(x[2::3])
        
    # Constraints
    # 1. Non-overlap: dist(i, j) >= r_i + r_j
    # 2. Boundary: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
    # 3. r_i >= 0
    
    constraints = []
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def non_overlap(x, i=i, j=j):
                xi = x[3*i]
                yi = x[3*i+1]
                ri = x[3*i+2]
                xj = x[3*j]
                yj = x[3*j+1]
                rj = x[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                # dist >= sum_r  =>  dist^2 >= sum_r^2 (since both positive)
                return dist_sq - sum_r**2
            constraints.append({'type': 'ineq', 'fun': non_overlap})
            
    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        # w - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[3*i] - x[3*i+2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        # h - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[3*i+1] - x[3*i+2]})
        
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, w)) # x
        bounds.append((0, h)) # y
        bounds.append((0, min(w, h)/2)) # r (upper bound is loose, constraints handle it)
        
    # Optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 100, 'ftol': 1e-9})
    
    if res.success:
        new_pos = np.zeros((n, 2))
        new_radii = np.zeros(n)
        for i in range(n):
            new_pos[i, 0] = res.x[3*i]
            new_pos[i, 1] = res.x[3*i+1]
            new_radii[i] = res.x[3*i+2]
        return new_pos, new_radii, np.sum(new_radii)
    else:
        return pos, radii, np.sum(radii)

def _sequential_place(pos, radii, w, h, start_index=0):
    """
    Sequentially place circles from start_index, finding best position for each.
    """
    for i in range(start_index, N):
        # Try to find a position for circle i that allows maximum radius
        best_r = 0.0
        best_pos = None
        
        # Try grid of candidate positions
        n_grid = 10
        for gx in range(n_grid):
            for gy in range(n_grid):
                cx = (gx + 0.5) * w / n_grid
                cy = (gy + 0.5) * h / n_grid
                
                # Check if this position is feasible (not overlapping existing circles)
                feasible = True
                for j in range(i):
                    dx = cx - pos[j, 0]
                    dy = cy - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < radii[j] + 0.01:  # small margin
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                pos[i] = [cx, cy]
                r_max = _max_radius_for_circle(i, pos, radii, w, h)
                if r_max > best_r:
                    best_r = r_max
                    best_pos = np.array([cx, cy])
        
        # Also try some random positions
        for _ in range(10):
            cx = np.random.rand() * w
            cy = np.random.rand() * h
            
            feasible = True
            for j in range(i):
                dx = cx - pos[j, 0]
                dy = cy - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < radii[j] + 0.01:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            pos[i] = [cx, cy]
            r_max = _max_radius_for_circle(i, pos, radii, w, h)
            if r_max > best_r:
                best_r = r_max
                best_pos = np.array([cx, cy])
        
        if best_pos is not None and best_r > 0.001:
            pos[i] = best_pos
            radii[i] = best_r
        else:
            # Fallback: place at center with tiny radius
            pos[i] = [w / 2, h / 2]
            radii[i] = 0.001
    
    return pos, radii

def _pack_for_aspect(w, h, start_time, deadline, num_trials=5):
    """
    Pack circles in w x h rectangle using sequential placement + refinement.
    Returns best (pos, radii) and score.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    for trial in range(num_trials):
        if time.time() > deadline - 10: # Reserve 10s for validation/final polish
            break
        
        # Initialize with grid + jitter
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            cols += 1
            rows = int(np.ceil(N / cols))
        
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            pos[i, 0] = (col + 0.5) * w / cols
            pos[i, 1] = (row + 0.5) * h / rows
        
        # Add jitter for different trials
        if trial > 0:
            jitter = 0.1 * min(w, h) / cols
            pos[:, 0] += np.random.randn(N) * jitter
            pos[:, 1] += np.random.randn(N) * jitter
            pos[:, 0] = np.clip(pos[:, 0], 0, w)
            pos[:, 1] = np.clip(pos[:, 1], 0, h)
        
        # Sequential placement to set initial radii
        radii[:] = 0.0
        pos, radii = _sequential_place(pos, radii, w, h)
        
        # Refine using SLSQP
        pos, radii, score = _refine_positions(pos, radii, w, h)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    return best_pos, best_radii, best_score

def _validate(pos, radii, w, h, tol=1e-7):
    """Check if configuration is valid (no overlaps, within bounds)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] < -tol or pos[i, 0] > w + tol:
            return False
        if pos[i, 1] < -tol or pos[i, 1] > h + tol:
            return False
        if radii[i] < -tol:
            return False
        # Wall check
        if radii[i] > pos[i, 0] + tol:
            return False
        if radii[i] > w - pos[i, 0] + tol:
            return False
        if radii[i] > pos[i, 1] + tol:
            return False
        if radii[i] > h - pos[i, 1] + tol:
            return False
    # Circle-circle check
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_validity(pos, radii, w, h):
    """Shrink radii slightly to ensure strict validity."""
    for _ in range(50):
        if _validate(pos, radii, w, h, tol=1e-8):
            return pos, radii, True
        radii *= 0.99
    return pos, radii, False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    np.random.seed(42)
    
    best_global_score = 0
    best_global_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline, num_trials=3)
        
        if pos is None:
            continue
        
        # Validate and fix
        pos, radii, valid = _fix_validity(pos, radii, w, h)
        
        if valid and score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side * 0.8
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val
        best_global_result = np.column_stack([pos, rad])
    
    return best_global_result
