import numpy as np
import time

N = 21
START = time.time()
DEADLINE = START + 145.0  # Safety margin

def _time_ok():
    return time.time() < DEADLINE

def _generate_initial(W, H, layout_type='grid'):
    """Generate initial positions and radii."""
    if layout_type == 'grid':
        # Use a 5x5 grid for 21 circles
        kx, ky = 5, 5
        xs = []
        ys = []
        for j in range(ky):
            for i in range(kx):
                if len(xs) >= N:
                    break
                x = (i + 0.5) * W / kx
                y = (j + 0.5) * H / ky
                xs.append(x)
                ys.append(y)
        while len(xs) < N:
            xs.append(W/2)
            ys.append(H/2)
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        # Start with very small radii to ensure no initial overlap
        r = np.ones(N) * min(W, H) / 20.0
        return np.column_stack([xs, ys, r])
    else:
        # Random initial positions
        xs = np.random.rand(N) * W
        ys = np.random.rand(N) * H
        r = np.ones(N) * min(W, H) / 20.0
        return np.column_stack([xs, ys, r])

def _compute_penalty_and_grad(x, W, H, penalty_weight):
    """
    Compute the penalized objective and its gradient.
    Objective: -sum(r) + penalty_weight * (sum of squared violations)
    Violations:
    1. Boundary: max(0, r - cx)^2, max(0, r - (W-cx))^2, etc.
    2. Overlap: max(0, (r_i + r_j)^2 - dist^2)^2
    """
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0
    
    # Boundary penalties
    # Left: cx - r < 0 => violation = r - cx
    viol_left = np.maximum(0, r - cx)
    obj += penalty_weight * np.sum(viol_left**2)
    grad[:, 0] -= penalty_weight * 2 * viol_left * 1.0
    grad[:, 2] -= penalty_weight * 2 * viol_left * (-1.0)
    
    # Right: W - cx - r < 0 => violation = r - (W - cx) = cx + r - W
    viol_right = np.maximum(0, cx + r - W)
    obj += penalty_weight * np.sum(viol_right**2)
    grad[:, 0] -= penalty_weight * 2 * viol_right * 1.0
    grad[:, 2] -= penalty_weight * 2 * viol_right * 1.0
    
    # Bottom: cy - r < 0 => violation = r - cy
    viol_bottom = np.maximum(0, r - cy)
    obj += penalty_weight * np.sum(viol_bottom**2)
    grad[:, 1] -= penalty_weight * 2 * viol_bottom * 1.0
    grad[:, 2] -= penalty_weight * 2 * viol_bottom * (-1.0)
    
    # Top: H - cy - r < 0 => violation = cy + r - H
    viol_top = np.maximum(0, cy + r - H)
    obj += penalty_weight * np.sum(viol_top**2)
    grad[:, 1] -= penalty_weight * 2 * viol_top * 1.0
    grad[:, 2] -= penalty_weight * 2 * viol_top * 1.0
    
    # Overlap penalties
    # For each pair (i, j), if dist < r_i + r_j, violation = (r_i + r_j)^2 - dist^2
    # We use squared violation for smoothness: ((r_i + r_j)^2 - dist^2)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist2 = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            sum_r2 = sum_r * sum_r
            
            # Only penalize if overlapping (or close to it)
            # To avoid numerical issues with sqrt, we work with squared distances
            # Violation v = max(0, sum_r2 - dist2)
            # Gradient of v^2: 2v * grad(v)
            # grad(v) w.r.t cx_i = -2*dx, w.r.t cy_i = -2*dy, w.r.t r_i = 2*sum_r
            # Note: if dist2 == 0, gradient is undefined, but we start with distinct points
            
            if dist2 < sum_r2:
                v = sum_r2 - dist2
                obj += penalty_weight * v**2
                
                # Gradient contribution
                g_val = penalty_weight * 2 * v
                grad[i, 0] -= g_val * (-2 * dx)
                grad[i, 1] -= g_val * (-2 * dy)
                grad[i, 2] -= g_val * (2 * sum_r)
                
                grad[j, 0] -= g_val * (2 * dx)
                grad[j, 1] -= g_val * (2 * dy)
                grad[j, 2] -= g_val * (2 * sum_r)
                
    return obj, grad

def _optimize_single(W, H, x0, max_iter=2000, step_size=0.01):
    """
    Gradient ascent on the negative penalized objective.
    """
    x = x0.copy()
    best_x = x.copy()
    best_obj = float('inf')
    
    # Adaptive penalty weight
    penalty_weight = 10.0
    step_size_current = step_size
    
    for it in range(max_iter):
        if not _time_ok():
            break
            
        obj, grad = _compute_penalty_and_grad(x, W, H, penalty_weight)
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
            
        # Simple gradient ascent with line search (or fixed step with decay)
        # We want to minimize obj, so we move in direction -grad
        # Normalize gradient to avoid large steps
        norm = np.linalg.norm(grad)
        if norm < 1e-6:
            break
            
        direction = -grad / norm
        
        # Line search: try a few step sizes
        found_step = False
        for k in range(10):
            s = step_size_current / (2 ** k)
            x_new = x + s * direction
            # Ensure bounds are roughly respected in the update (soft)
            # Clip positions to [0, W] and [0, H]
            x_new[:, 0] = np.clip(x_new[:, 0], 0, W)
            x_new[:, 1] = np.clip(x_new[:, 1], 0, H)
            x_new[:, 2] = np.clip(x_new[:, 2], 0, min(W, H)/2)
            
            obj_new, _ = _compute_penalty_and_grad(x_new, W, H, penalty_weight)
            
            if obj_new < obj:
                x = x_new
                found_step = True
                break
        
        if not found_step:
            # If no step improved, reduce step size and penalty weight slightly
            step_size_current *= 0.5
            penalty_weight *= 0.9
            if step_size_current < 1e-6:
                break
                
    return best_x, -best_obj # Return score as sum of radii (approx, since penalty is included in obj)

def _validate_and_fix(x, W, H):
    """
    Ensure the solution is strictly valid.
    Returns (valid, x_fixed, score)
    """
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    # Clip positions
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    r = np.clip(r, 0, min(W, H)/2)
    
    # Check boundaries
    # If a circle is too close to a boundary, shrink it
    for i in range(N):
        min_dist = min(cx[i], W - cx[i], cy[i], H - cy[i])
        if r[i] > min_dist:
            r[i] = min_dist * 0.99
    
    # Check overlaps
    # If circles overlap, shrink the smaller one or both
    # Simple iterative fix
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                if dist < sum_r:
                    # Shrink both slightly
                    # Proportional to radius
                    total_r = sum_r
                    r[i] *= (dist / total_r) * 0.99
                    r[j] *= (dist / total_r) * 0.99
                    changed = True
        if not changed:
            break
            
    # Final check
    valid = True
    for i in range(N):
        if r[i] < 0:
            valid = False
            break
        if cx[i] < r[i] - 1e-9 or cx[i] > W - r[i] + 1e-9:
            valid = False
            break
        if cy[i] < r[i] - 1e-9 or cy[i] > H - r[i] + 1e-9:
            valid = False
            break
            
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist2 = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist2 < sum_r*sum_r - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
    score = np.sum(r)
    if valid:
        return True, np.column_stack([cx, cy, r]), score
    else:
        return False, None, 0.0

def construct():
    best_score = 0.0
    best_out = None
    
    # Try a few aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33]
    
    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        
        # Try multiple initial layouts
        layouts = ['grid', 'random']
        
        for layout in layouts:
            if not _time_ok():
                break
                
            # Try a few random seeds for 'random' layout
            num_starts = 3 if layout == 'random' else 1
            
            for k in range(num_starts):
                if not _time_ok():
                    break
                    
                x0 = _generate_initial(W, H, layout)
                
                # Run optimization
                x_opt, _ = _optimize_single(W, H, x0)
                
                # Validate and fix
                valid, x_fixed, score = _validate_and_fix(x_opt, W, H)
                
                if valid and score > best_score:
                    best_score = score
                    best_out = x_fixed
                    
    # Fallback
    if best_out is None:
        W, H = 1.0, 1.0
        g = _generate_initial(W, H, 'grid')
        # Force validity for fallback
        _, g_fixed, _ = _validate_and_fix(g, W, H)
        if g_fixed is not None:
            best_out = g_fixed
        else:
            # Last resort: tiny circles
            best_out = np.column_stack([
                np.random.rand(N) * 0.9 + 0.05,
                np.random.rand(N) * 0.9 + 0.05,
                np.ones(N) * 0.01
            ])
            
    return best_out
