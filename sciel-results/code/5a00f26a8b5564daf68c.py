"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 21
DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _rect_perimeter(w, h):
    return 2.0 * (w + h)

def _validate(arr, tol=1e-9):
    """Check if arrangement is valid: disjoint interiors, inside rectangle, perimeter <= 4."""
    if len(arr) != N:
        return False
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    
    # All radii positive
    if np.any(rs <= 0):
        return False
    
    # Find bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w <= 0 or h <= 0:
        return False
    
    # Perimeter constraint
    if 2.0 * (w + h) > 4.0 + tol:
        return False
    
    # Check disjointness (interiors disjoint means distance >= r_i + r_j)
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True

def _objective(arr):
    """Sum of radii."""
    return np.sum(arr[:, 2])

def _grid_initial(n=21):
    """Create a grid-based initial arrangement."""
    # Try different grid configurations
    best_arr = None
    best_val = -1
    
    # 5x5 grid with 21 circles
    for cols in range(4, 7):
        rows = (n + cols - 1) // cols
        for start_shift in [0.0, 0.5 / cols]:
            cells = []
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= n:
                        break
                    x = (j + 0.5) / cols
                    y = (i + 0.5) / rows
                    r = min(0.5 / cols, 0.5 / rows) - 1e-4
                    cells.append([x, y, r])
                    count += 1
                if count >= n:
                    break
            
            arr = np.array(cells)
            if _validate(arr):
                val = _objective(arr)
                if val > best_val:
                    best_val = val
                    best_arr = arr.copy()
    
    # Also try 7x3 grid
    for cols, rows in [(7, 3), (3, 7), (6, 4), (4, 6), (5, 5)]:
        cells = []
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                r = min(0.5 / cols, 0.5 / rows) - 1e-4
                cells.append([x, y, r])
                count += 1
            if count >= n:
                break
        arr = np.array(cells)
        if _validate(arr):
            val = _objective(arr)
            if val > best_val:
                best_val = val
                best_arr = arr.copy()
    
    if best_arr is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr

def _local_optimize(arr, max_iter=500, lr_start=0.01, lr_end=0.0001, tol=1e-8):
    """
    Gradient-free local optimization using a simple approach.
    We use a penalty method: maximize sum(r_i) subject to constraints.
    Use coordinate descent or a simple gradient step.
    """
    arr = arr.copy()
    n = len(arr)
    
    # Normalize to fit in a box of perimeter 4
    # First, scale to fit
    def _normalize(arr):
        xs = arr[:, 0]
        ys = arr[:, 1]
        rs = arr[:, 2]
        
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        
        w = max_x - min_x
        h = max_y - min_y
        
        if w <= 0 or h <= 0:
            return arr
        
        # Scale to perimeter 4
        scale = 4.0 / (2.0 * (w + h))
        if scale > 1.0:
            scale = 1.0  # Don't expand
        
        # Center
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        new_arr = arr.copy()
        new_arr[:, 0] = (arr[:, 0] - cx) * scale + 1.0
        new_arr[:, 1] = (arr[:, 1] - cy) * scale + 1.0
        new_arr[:, 2] = arr[:, 2] * scale
        
        return new_arr
    
    arr = _normalize(arr)
    
    # Simple iterative refinement
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        improved = False
        lr = lr_start * (lr_end / lr_start) ** (it / max_iter)
        
        # Try to increase each radius slightly and adjust positions
        for i in range(n):
            # Try increasing radius
            old_r = arr[i, 2]
            new_r = old_r * (1 + lr * 0.1)
            
            # Check if we can fit with new radius
            # Move circle away from others
            feasible = True
            for j in range(n):
                if i == j:
                    continue
                dx = arr[i, 0] - arr[j, 0]
                dy = arr[i, 1] - arr[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = new_r + arr[j, 2]
                if dist < min_dist:
                    # Push apart
                    if dist > 1e-10:
                        push = (min_dist - dist) / dist
                        arr[i, 0] += dx * push * 0.5
                        arr[i, 1] += dy * push * 0.5
                        arr[j, 0] -= dx * push * 0.5
                        arr[j, 1] -= dy * push * 0.5
                    else:
                        arr[i, 0] += lr * 0.1
                    feasible = False
                    break
            
            if feasible:
                arr[i, 2] = new_r
                improved = True
        
        # Normalize
        arr = _normalize(arr)
        
        if not improved:
            break
    
    return arr

def _simulated_annealing(arr, max_iter=2000, t_start=0.05, t_end=0.001):
    """Simple simulated annealing for improvement."""
    arr = arr.copy()
    n = len(arr)
    best_arr = arr.copy()
    best_val = _objective(arr)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Random perturbation
        new_arr = arr.copy()
        idx = random.randint(0, n - 1)
        delta_x = random.gauss(0, t)
        delta_y = random.gauss(0, t)
        delta_r = random.gauss(0, t * 0.5)
        
        new_arr[idx, 0] += delta_x
        new_arr[idx, 1] += delta_y
        new_arr[idx, 2] = max(0.01, new_arr[idx, 2] + delta_r)
        
        # Check validity
        if _validate(new_arr):
            new_val = _objective(new_arr)
            if new_val > best_val:
                best_val = new_val
                best_arr = new_arr.copy()
            # Accept with probability
            if new_val >= _objective(arr) or random.random() < np.exp((new_val - _objective(arr)) / max(t, 1e-10)):
                arr = new_arr
        else:
            # Reject invalid moves
            pass
    
    return best_arr

def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    candidates = []
    
    # Grid-based initial
    arr_grid = _grid_initial(N)
    candidates.append(arr_grid)
    
    # Try different aspect ratios
    for aspect in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        if time.time() > DEADLINE - 10:
            break
        
        # Create circles in a rectangle with given aspect ratio
        # Perimeter = 2(w+h) = 4, so w+h = 2
        # If aspect = w/h, then w = aspect*h, so h(aspect+1) = 2, h = 2/(aspect+1)
        h = 2.0 / (aspect + 1)
        w = aspect * h
        
        # Place circles in a grid within this rectangle
        # Estimate number of circles per row/col
        # Average circle diameter ~ 2r, want to fit in w x h
        # Try different grid layouts
        for cols in range(3, 8):
            rows = (N + cols - 1) // cols
            if rows < 1:
                continue
            
            # Circle radius estimate
            r_est = min(w / cols, h / rows) * 0.45
            
            cells = []
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= N:
                        break
                    x = (j + 0.5) * w / cols
                    y = (i + 0.5) * h / rows
                    r = min(w / cols, h / rows) * 0.45 - 1e-4
                    cells.append([x + 0.1, y + 0.1, r])  # Offset to be in positive quadrant
                    count += 1
                if count >= N:
                    break
            
            if len(cells) == N:
                arr = np.array(cells)
                # Scale to fit perimeter 4
                min_x = np.min(arr[:, 0] - arr[:, 2])
                max_x = np.max(arr[:, 0] + arr[:, 2])
                min_y = np.min(arr[:, 1] - arr[:, 2])
                max_y = np.max(arr[:, 1] + arr[:, 2])
                w_actual = max_x - min_x
                h_actual = max_y - min_y
                scale = 4.0 / (2.0 * (w_actual + h_actual))
                arr[:, 0] = (arr[:, 0] - min_x) * scale + 0.1
                arr[:, 1] = (arr[:, 1] - min_y) * scale + 0.1
                arr[:, 2] = arr[:, 2] * scale
                
                if _validate(arr):
                    candidates.append(arr)
    
    # Optimize each candidate
    best_arr = None
    best_val = -1
    
    for arr in candidates:
        if time.time() > DEADLINE - 10:
            break
        
        # Local optimization
        opt_arr = _local_optimize(arr, max_iter=200)
        
        if _validate(opt_arr):
            val = _objective(opt_arr)
            if val > best_val:
                best_val = val
                best_arr = opt_arr.copy()
        
        # Try simulated annealing on the best so far
        if best_arr is not None and time.time() < DEADLINE - 20:
            sa_arr = _simulated_annealing(best_arr, max_iter=1000)
            if _validate(sa_arr):
                val = _objective(sa_arr)
                if val > best_val:
                    best_val = val
                    best_arr = sa_arr.copy()
    
    if best_arr is None:
        # Fallback
        best_arr = arr_grid
    
    # Final validation
    if not _validate(best_arr):
        # Try to fix
        best_arr = _local_optimize(best_arr, max_iter=100)
        if not _validate(best_arr):
            # Last resort: simple grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                     for i in range(side) for j in range(side)]
            best_arr = np.array(cells[:N])
    
    return best_arr

# EVOLVE-BLOCK-END
