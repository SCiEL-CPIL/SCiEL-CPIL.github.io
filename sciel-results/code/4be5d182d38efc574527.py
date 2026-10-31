"""Circle packing in perimeter-4 rectangle: Soft-Penalty Gradient Descent with Annealing."""
import numpy as np
import time
from scipy.optimize import minimize

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

def _compute_penalty_and_grad(x, y, r, W, H, penalty_weight=100.0):
    """
    Compute the objective value and gradient for a soft-penalty formulation.
    Objective: Maximize sum(r) - penalty_weight * (overlap_penalty + wall_penalty)
    We minimize the negative of this.
    
    Returns:
        value: The objective value to minimize (negative sum r + penalties)
        grad: Gradient of the objective w.r.t [x, y, r, W, H]
    """
    # Note: We treat W and H as fixed in the inner loop for speed, 
    # optimizing aspect ratio separately or jointly.
    # Here we optimize x, y, r jointly. W, H are parameters.
    
    # 1. Overlap Penalty
    # For each pair (i, j), if dist < r_i + r_j, penalty = (r_i + r_j - dist)^2
    # Gradient w.r.t x_i: -2 * (r_i + r_j - dist) * (x_i - x_j) / dist
    # Gradient w.r.t r_i: -2 * (r_i + r_j - dist)
    
    overlap_pen = 0.0
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    grad_r = np.zeros(N)
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-9:
                # Avoid division by zero, push apart randomly
                dx = 1e-6
                dy = 1e-6
                d = np.sqrt(dx*dx + dy*dy)
            
            dist_diff = r[i] + r[j] - d
            if dist_diff > 0:
                # Penalty is dist_diff^2
                overlap_pen += dist_diff * dist_diff
                
                # Gradient of (dist_diff)^2 w.r.t x_i
                # d/dx_i (r_i + r_j - dist)^2 = 2*(r_i + r_j - dist) * (-d(dist)/dx_i)
                # d(dist)/dx_i = (x_i - x_j) / dist
                # So term is -2 * dist_diff * (x_i - x_j) / dist
                factor = -2.0 * dist_diff / d
                grad_x[i] += factor * dx
                grad_y[i] += factor * dy
                grad_x[j] -= factor * dx
                grad_y[j] -= factor * dy
                
                # Gradient w.r.t r_i
                # d/d r_i (r_i + r_j - dist)^2 = 2 * (r_i + r_j - dist)
                grad_r[i] += 2.0 * dist_diff
                grad_r[j] += 2.0 * dist_diff
    
    # 2. Wall Penalty
    # Penalty if x_i < r_i: (r_i - x_i)^2
    # Penalty if x_i > W - r_i: (x_i - (W - r_i))^2
    # Similar for y.
    
    wall_pen = 0.0
    
    # Left wall: x_i - r_i < 0
    for i in range(N):
        # Left
        diff_l = r[i] - x[i]
        if diff_l > 0:
            wall_pen += diff_l * diff_l
            grad_x[i] += -2.0 * diff_l  # d/dx_i (r_i - x_i)^2 = -2(r_i - x_i)
            grad_r[i] += 2.0 * diff_l   # d/dr_i (r_i - x_i)^2 = 2(r_i - x_i)
        
        # Right
        diff_r = x[i] - (W - r[i])
        if diff_r > 0:
            wall_pen += diff_r * diff_r
            grad_x[i] += 2.0 * diff_r
            grad_r[i] += -2.0 * diff_r
        
        # Bottom
        diff_b = r[i] - y[i]
        if diff_b > 0:
            wall_pen += diff_b * diff_b
            grad_y[i] += -2.0 * diff_b
            grad_r[i] += 2.0 * diff_b
        
        # Top
        diff_t = y[i] - (H - r[i])
        if diff_t > 0:
            wall_pen += diff_t * diff_t
            grad_y[i] += 2.0 * diff_t
            grad_r[i] += -2.0 * diff_t
    
    total_penalty = penalty_weight * (overlap_pen + wall_pen)
    
    # Objective to minimize: -sum(r) + total_penalty
    obj_val = -np.sum(r) + total_penalty
    
    # Gradient of objective
    grad_obj_x = grad_x
    grad_obj_y = grad_y
    grad_obj_r = grad_r - np.ones(N) # -1 from -sum(r)
    
    # Combine gradients
    grad = np.concatenate([grad_obj_x, grad_obj_y, grad_obj_r])
    
    return obj_val, grad

def _polish_radii(x, y, W, H, r_init):
    """
    Given fixed positions and box, maximize radii.
    Iterative projection is robust and fast for small N.
    """
    r = r_init.copy()
    # Precompute distances
    d_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_mat[i,j] = np.sqrt(dx*dx + dy*dy)
                
    # Wall limits
    wall_lim = np.minimum(np.minimum(x, W-x), np.minimum(y, H-y))
    
    for _ in range(100):
        improved = False
        for i in range(N):
            # Max radius allowed by walls
            max_r = wall_lim[i]
            
            # Max radius allowed by neighbors
            for j in range(N):
                if i != j:
                    # r_i <= d_ij - r_j
                    allowed = d_mat[i,j] - r[j]
                    if allowed < max_r:
                        max_r = allowed
                        
            if max_r < 0:
                max_r = 1e-9 # Should not happen if valid, but safety
                
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                improved = True
        if not improved:
            break
            
    return r

def _soft_optimize(x0, y0, r0, W, H, max_iters=2000, lr_init=0.05, lr_decay=0.99, penalty_weight=50.0, time_limit=10.0):
    """
    Run gradient descent on the soft-penalty objective.
    """
    start_time = time.time()
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()
    
    # Ensure initial validity roughly by clipping
    r = np.maximum(r, 1e-6)
    
    best_val = np.inf
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute penalty and gradient
        val, grad = _compute_penalty_and_grad(x, y, r, W, H, penalty_weight)
        
        if val < best_val:
            best_val = val
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
        # Adaptive learning rate based on penalty magnitude
        # If penalty is high, we need larger steps to escape
        # If penalty is low, we need smaller steps for precision
        penalty_part = val + np.sum(r)
        if penalty_part > 1.0:
            lr = lr_init
        else:
            lr = lr_init * (0.1 + 0.9 * (1.0 - penalty_part))
            
        # Gradient Descent Step
        grad_x, grad_y, grad_r = np.split(grad, 3)
        
        x_new = x - lr * grad_x
        y_new = y - lr * grad_y
        r_new = r - lr * grad_r
        
        # Enforce basic bounds to prevent divergence
        r_new = np.maximum(r_new, 1e-6)
        x_new = np.clip(x_new, 0.01, W - 0.01)
        y_new = np.clip(y_new, 0.01, H - 0.01)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite([x_new, y_new, r_new])):
            break
            
        x, y, r = x_new, y_new, r_new
        
        # Decay learning rate
        lr_init *= lr_decay
        
    return best_x, best_y, best_r, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, W, H):
        nonlocal best_value, best_result
        # Validate
        if _is_valid(x, y, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), W, H)
                
    # Strategy:
    # 1. Iterate over aspect ratios.
    # 2. For each aspect ratio, generate multiple random initializations (hex grid + noise).
    # 3. Run soft optimization from each.
    # 4. Polish the best results.
    
    # Aspect ratios: W + H = 2. W in [0.5, 1.5]
    aspect_ws = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Number of restarts per aspect ratio
    restarts_per_aspect = 5
    
    total_restarts = len(aspect_ws) * restarts_per_aspect
    
    for W in aspect_ws:
        if time.time() > deadline - 20:
            break
            
        H = 2.0 - W
        
        # Skip if dimensions are too small
        if W < 0.3 or H < 0.3:
            continue
            
        for restart in range(restarts_per_aspect):
            if time.time() > deadline - 10:
                break
                
            # Generate initial positions
            # Use a hexagonal-like grid with random noise
            rng = np.random.default_rng(int(time.time()*1000) + restart + int(W*100))
            
            # Estimate radius
            r_est = np.sqrt((W * H) / (N * 2 * np.sqrt(3)))
            
            # Grid placement
            # 5x5 grid is good for 21 circles (25 spots, remove 4)
            cols = 5
            rows = 5
            xs = []
            ys = []
            for i in range(cols):
                for j in range(rows):
                    cx = (i + 0.5) * (W / cols)
                    cy = (j + 0.5) * (H / rows)
                    # Add hex offset for odd rows
                    if j % 2 == 1:
                        cx += 0.25 * (W / cols)
                    xs.append(cx)
                    ys.append(cy)
            
            # We have 25 positions. Pick 21 randomly or remove corners?
            # Removing corners allows larger circles in the center?
            # Let's just pick the first 21 and add noise.
            x0 = np.array(xs[:21])
            y0 = np.array(ys[:21])
            
            # Add significant noise to break symmetry
            noise_scale = 0.1 * r_est
            x0 = x0 + rng.uniform(-noise_scale, noise_scale, N)
            y0 = y0 + rng.uniform(-noise_scale, noise_scale, N)
            
            # Clip to box
            x0 = np.clip(x0, r_est*0.5, W - r_est*0.5)
            y0 = np.clip(y0, r_est*0.5, H - r_est*0.5)
            
            # Initial radii
            r0 = np.full(N, r_est * 0.7)
            
            # Run soft optimization
            try:
                x_res, y_res, r_res, val = _soft_optimize(
                    x0, y0, r0, W, H,
                    max_iters=1000,
                    lr_init=0.02,
                    lr_decay=0.995,
                    penalty_weight=100.0,
                    time_limit=5.0
                )
                
                # Polish radii
                r_polished = _polish_radii(x_res, y_res, W, H, r_res)
                
                update_best(x_res, y_res, r_polished, W, H)
                
            except Exception as e:
                pass # Ignore failures

    # Fallback if no valid result found
    if best_result is None:
        W, H = 1.0, 1.0
        # Simple grid
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
        r = np.full(N, 0.1)
        best_result = (x, y, r, W, H)
        
    x, y, r, W, H = best_result
    
    # Final safety shrink
    for _ in range(3):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
