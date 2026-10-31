"""Circle packing in perimeter-4 rectangle: Discrete Simulated Annealing."""
import numpy as np
import time
import random

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _maximize_radii(x, y, W, H):
    """
    Given fixed positions x, y and box W, H, find the maximum radii r
    such that circles are disjoint and inside the box.
    This is a linear programming problem:
    Maximize sum(r_i)
    Subject to:
      r_i >= 0
      r_i <= x_i
      r_i <= W - x_i
      r_i <= y_i
      r_i <= H - y_i
      r_i + r_j <= dist(i, j) for all i < j
    
    We solve this using a simple iterative projection method (Kaczmarz-like)
    which converges quickly for small N.
    """
    r = np.zeros(N)
    
    # Precompute pairwise distances
    d_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_mat[i, j] = np.sqrt(dx*dx + dy*dy)
                
    # Wall limits
    wall_lim = np.minimum(np.minimum(x, W-x), np.minimum(y, H-y))
    
    # Initialize r to wall limits (or slightly less to be safe)
    r = np.maximum(wall_lim, 0.0)
    
    # Iterative refinement
    # We want to satisfy r_i + r_j <= d_ij and r_i <= wall_lim_i
    # This is equivalent to finding the largest r in the intersection of half-spaces.
    # Since it's a small LP, we can also just use scipy.optimize.linprog if available,
    # but iterative projection is dependency-light and fast enough for N=21.
    
    for _ in range(500):
        changed = False
        # Update each circle's radius to be the minimum of its constraints
        for i in range(N):
            # Constraint from walls
            max_r = wall_lim[i]
            
            # Constraint from other circles
            for j in range(N):
                if i != j:
                    # r_i <= d_ij - r_j
                    allowed = d_mat[i, j] - r[j]
                    if allowed < max_r:
                        max_r = allowed
                        
            if max_r < 0:
                max_r = 0.0
                
            if abs(r[i] - max_r) > 1e-10:
                r[i] = max_r
                changed = True
                
        if not changed:
            break
            
    return r

def _hexagonal_init(W, H):
    """Generate a hexagonal close-packed layout for N circles in W x H box."""
    x = np.zeros(N)
    y = np.zeros(N)
    
    # Estimate radius
    area = W * H
    # Hexagonal packing density is pi / (2 * sqrt(3)) ~ 0.9069
    # Area per circle ~ r^2 * 2 * sqrt(3)
    r_est = np.sqrt(area / (N * 2 * np.sqrt(3)))
    
    # Determine grid size
    # We want roughly sqrt(N) rows and cols
    cols = int(np.ceil(np.sqrt(N * W / H)))
    rows = int(np.ceil(N / cols))
    
    # Spacing
    dx = W / (cols + 0.5)
    dy = H / (rows + 0.5)
    
    # Adjust for hexagonal offset
    # Actually, let's just place them in a grid with hex offset
    # and let the optimizer fix it.
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= N:
                break
            offset = 0.5 * (row % 2)
            x[idx] = (col + 0.5 + offset) * dx
            y[idx] = (row + 0.5) * dy
            idx += 1
            
    # Ensure within bounds
    r_init = r_est * 0.8
    x = np.clip(x, r_init, W - r_init)
    y = np.clip(y, r_init, H - r_init)
    
    return x, y

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_value = 0.0
    best_result = None
    best_x, best_y, best_r, best_W, best_H = None, None, None, 1.0, 1.0
    
    def update_best(x, y, r, W, H):
        nonlocal best_value, best_result, best_x, best_y, best_r, best_W, best_H
        if _is_valid(x, y, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), W, H)
                best_x, best_y, best_r, best_W, best_H = x.copy(), y.copy(), r.copy(), W, H
                
    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature
    alpha = 0.999  # Cooling rate
    T_min = 1e-6
    
    # We will run a few independent SA runs with different aspect ratios
    aspect_deltas = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
    
    for delta in aspect_deltas:
        if time.time() > deadline - 10:
            break
            
        W = 1.0 + delta
        H = 1.0 - delta
        
        if W < 0.3 or H < 0.3:
            continue
            
        # Initialize
        x, y = _hexagonal_init(W, H)
        r = _maximize_radii(x, y, W, H)
        current_value = np.sum(r)
        
        update_best(x, y, r, W, H)
        
        # Annealing loop
        t_curr = T
        step_size = 0.05
        
        while t_curr > T_min and time.time() < deadline - 5:
            # Propose a move
            # 1. Move a random circle
            i = random.randint(0, N - 1)
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            
            x_new = x.copy()
            y_new = y.copy()
            x_new[i] += dx
            y_new[i] += dy
            
            # Clip to box (with current radius as margin to avoid immediate invalidity? 
            # No, better to clip to box bounds [0, W] and [0, H] and let radius calc handle it)
            # Actually, if we clip to [0, W], the circle might go out of bounds if r > 0.
            # But _maximize_radii will set r=0 if it's out of bounds? No, it sets r based on distance to walls.
            # If x_new[i] < 0, wall_lim will be negative, so r becomes 0.
            # This is a valid state (circle shrinks to point), but we want to avoid it.
            # Let's clip to [r_i, W-r_i] if possible, otherwise to [0, W].
            
            # Simple clip to box
            x_new[i] = np.clip(x_new[i], 0, W)
            y_new[i] = np.clip(y_new[i], 0, H)
            
            # Calculate new radii
            r_new = _maximize_radii(x_new, y_new, W, H)
            new_value = np.sum(r_new)
            
            delta_val = new_value - current_value
            
            # Acceptance criterion
            if delta_val > 0 or random.random() < np.exp(delta_val / t_curr):
                x, y, r = x_new, y_new, r_new
                current_value = new_value
                update_best(x, y, r, W, H)
                
            # Cool down
            t_curr *= alpha
            # Slightly increase step size as we cool? Or decrease?
            # Usually decrease to settle.
            # But we want to explore. Let's keep step size constant or decay slowly.
            # step_size *= 0.999 
            
        # After SA, do a local polish (gradient-free local search)
        # Try small perturbations to each circle
        if best_result is not None and time.time() < deadline - 2:
             # Use the best result found so far globally, not just this aspect ratio
             # Actually, let's polish the best result from this run if it's better than global best?
             # The global best is tracked. Let's just return the global best.
             pass
             
    # Final safety: if no result, return a trivial valid one
    if best_result is None:
        W, H = 1.0, 1.0
        x = np.linspace(0.1, 0.9, N)
        y = np.full(N, 0.5)
        r = np.full(N, 0.01)
        # This might not be valid if they overlap.
        # Let's construct a valid trivial packing:
        # Place circles in a grid with very small radius.
        cols = 5
        rows = 5
        xs = []
        ys = []
        for i in range(5):
            for j in range(5):
                cx = (i + 0.5) * (W / 5.0)
                cy = (j + 0.5) * (H / 5.0)
                xs.append(cx)
                ys.append(cy)
        x = np.array(xs[:21])
        y = np.array(ys[:21])
        r = np.full(N, 0.05)
        best_result = (x, y, r, W, H)
        
    x, y, r, W, H = best_result
    
    # Final validation and slight shrink if necessary to ensure strict validity
    if not _is_valid(x, y, r, W, H):
        # Shrink radii until valid
        scale = 1.0
        for _ in range(100):
            r_new = r * scale
            if _is_valid(x, y, r_new, W, H):
                r = r_new
                break
            scale *= 0.9
        else:
            # Fallback to tiny circles
            r = np.full(N, 0.01)
            
    return np.column_stack((x, y, r))
