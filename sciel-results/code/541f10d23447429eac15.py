import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _objective_and_grad(z, w, h, penalty_weight=100.0):
    """
    z: array of shape (3*N,) containing [x1, y1, r1, ..., xN, yN, rN]
    Returns: (objective_value, gradient)
    Objective: -sum(r) + penalty_weight * (wall_violations + circle_violations)
    """
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Objective part 1: -sum(r)
    obj = -np.sum(r)
    grad = np.zeros_like(z)
    grad[2::3] = -1.0
    
    # Penalty part: Wall violations
    # r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
    # Violation v = max(0, r - dist_to_wall)
    # Smooth approximation or hard max? Hard max is fine for L-BFGS-B if we are careful, 
    # but let's use a quadratic penalty on the violation magnitude.
    
    # Calculate distances to walls
    dx_min = x
    dx_max = w - x
    dy_min = y
    dy_max = h - y
    
    # Violations
    v_x_min = np.maximum(0, r - dx_min)
    v_x_max = np.maximum(0, r - dx_max)
    v_y_min = np.maximum(0, r - dy_min)
    v_y_max = np.maximum(0, r - dy_max)
    
    wall_pen = np.sum(v_x_min**2 + v_x_max**2 + v_y_min**2 + v_y_max**2)
    obj += penalty_weight * wall_pen
    
    # Gradients for wall penalties
    # d/dx (r-x)^2 = 2(r-x)(-1) = 2(x-r) if r>x else 0
    # d/dr (r-x)^2 = 2(r-x)(1) = 2(r-x) if r>x else 0
    
    mask_x_min = v_x_min > 0
    mask_x_max = v_x_max > 0
    mask_y_min = v_y_min > 0
    mask_y_max = v_y_max > 0
    
    grad[0::3][mask_x_min] += penalty_weight * 2 * (x[mask_x_min] - r[mask_x_min])
    grad[2::3][mask_x_min] += penalty_weight * 2 * (r[mask_x_min] - x[mask_x_min])
    
    grad[0::3][mask_x_max] += penalty_weight * 2 * (x[mask_x_max] - (w - x[mask_x_max])) # d/dx (r-(w-x))^2 = 2(r-(w-x))(1) -> wait.
    # Let u = r - (w-x) = r - w + x. du/dx = 1. du/dr = 1.
    # Penalty term: u^2. d/dx u^2 = 2u * 1 = 2(r-w+x).
    # So grad_x += 2(r - w + x).
    # My previous line: 2 * (x - (w-x)) = 2(2x-w). That's wrong.
    # Correct: grad_x += penalty_weight * 2 * (r - w + x)
    # Let's re-verify. v = max(0, r - (w-x)). If active, v = r - w + x.
    # d/dx (r-w+x)^2 = 2(r-w+x).
    # d/dr (r-w+x)^2 = 2(r-w+x).
    
    # Let's just compute the active violation values and apply gradients directly.
    
    # Reset gradients for wall parts to be safe and clear
    grad[0::3] = -0.0 # Start fresh for x, y (r part already set to -1)
    grad[1::3] = -0.0
    grad[2::3] = -1.0
    
    # Recalculate and apply wall gradients
    # v_x_min = max(0, r - x)
    # if v > 0: grad_x += 2*penalty*v*(-1) = -2*penalty*(r-x); grad_r += 2*penalty*v*(1) = 2*penalty*(r-x)
    # v_x_max = max(0, r - (w-x)) = max(0, r - w + x)
    # if v > 0: grad_x += 2*penalty*v*(1) = 2*penalty*(r-w+x); grad_r += 2*penalty*v*(1) = 2*penalty*(r-w+x)
    
    vals_x_min = np.maximum(0, r - x)
    vals_x_max = np.maximum(0, r - (w - x))
    vals_y_min = np.maximum(0, r - y)
    vals_y_max = np.maximum(0, r - (h - y))
    
    # x gradients
    grad[0::3] -= penalty_weight * 2 * vals_x_min
    grad[0::3] += penalty_weight * 2 * vals_x_max
    # y gradients
    grad[1::3] -= penalty_weight * 2 * vals_y_min
    grad[1::3] += penalty_weight * 2 * vals_y_max
    # r gradients
    grad[2::3] += penalty_weight * 2 * (vals_x_min + vals_x_max + vals_y_min + vals_y_max)
    
    # Penalty part: Circle overlaps
    # v_ij = max(0, r_i + r_j - d_ij)
    # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    
    # To avoid sqrt(0), add small epsilon or handle carefully
    # We will loop over pairs for gradient computation as vectorizing is complex with max
    
    circle_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq)
            
            r_sum = r[i] + r[j]
            
            if d < 1e-12:
                # Overlap completely, push apart arbitrarily or just penalize
                v = r_sum
                # Gradient undefined at 0, use small perturbation direction or skip?
                # If d=0, we are in deep overlap. 
                # Let's just add a large penalty and skip gradient to avoid NaNs, 
                # or use a random direction.
                circle_pen += v*v
                continue
            
            if d < r_sum:
                v = r_sum - d
                circle_pen += v*v
                
                # Gradients
                # d/dx_i (r_i + r_j - d)^2 = 2v * (-dd/dx_i) = 2v * (-(x_i-x_j)/d)
                # dd/dx_i = (x_i-x_j)/d
                # So grad_x_i += -2*penalty*v*(x_i-x_j)/d
                # grad_x_j += -2*penalty*v*(x_j-x_i)/d = 2*penalty*v*(x_i-x_j)/d
                
                coeff = penalty_weight * 2 * v / d
                
                grad[3*i + 0] -= coeff * dx
                grad[3*j + 0] += coeff * dx
                
                grad[3*i + 1] -= coeff * dy
                grad[3*j + 1] += coeff * dy
                
                # d/dr_i = 2v * 1
                # d/dr_j = 2v * 1
                grad[3*i + 2] += penalty_weight * 2 * v
                grad[3*j + 2] += penalty_weight * 2 * v
    
    obj += penalty_weight * circle_pen
    
    return obj, grad

def _is_valid(pos, radii, w, h, tol=1e-7):
    """Check validity of the configuration."""
    # Check bounds
    for i in range(N):
        if pos[i, 0] < -tol or pos[i, 0] > w + tol:
            return False
        if pos[i, 1] < -tol or pos[i, 1] > h + tol:
            return False
        if radii[i] < -tol:
            return False
        # Wall clearance
        if radii[i] > pos[i, 0] + tol:
            return False
        if radii[i] > w - pos[i, 0] + tol:
            return False
        if radii[i] > pos[i, 1] + tol:
            return False
        if radii[i] > h - pos[i, 1] + tol:
            return False
            
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _pack_for_aspect(w, h, start_time, deadline, num_trials=10):
    """
    Pack circles in w x h rectangle using gradient-based optimization.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    # Bounds for L-BFGS-B
    # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
    r_max = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.0, r_max))
        
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        # Initialization: Grid + Jitter
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            cols += 1
            rows = int(np.ceil(N / cols))
            
        z0 = np.zeros(3 * N)
        for i in range(N):
            col = i % cols
            row = i // cols
            z0[3*i + 0] = (col + 0.5) * w / cols
            z0[3*i + 1] = (row + 0.5) * h / rows
            # Initial radius: small, or estimate based on grid spacing
            # Estimate max possible radius in grid cell
            cell_w = w / cols
            cell_h = h / rows
            z0[3*i + 2] = min(cell_w, cell_h) / 2.0 * 0.9 # 90% of half-cell size
            
        # Add jitter for diversity
        if trial > 0:
            jitter_scale = 0.1 * min(w, h)
            z0[0::3] += np.random.randn(N) * jitter_scale
            z0[1::3] += np.random.randn(N) * jitter_scale
            # Clip positions to bounds
            z0[0::3] = np.clip(z0[0::3], 0, w)
            z0[1::3] = np.clip(z0[1::3], 0, h)
            # Ensure radii are not negative
            z0[2::3] = np.maximum(0, z0[2::3])
            
        # Optimize
        try:
            res = minimize(_objective_and_grad, z0, args=(w, h, 100.0), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            if res.success or res.fun < 0: # Check if we got a good result
                z_opt = res.x
                pos = np.column_stack((z_opt[0::3], z_opt[1::3]))
                radii = z_opt[2::3]
                
                # Check validity
                if _is_valid(pos, radii, w, h):
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    np.random.seed(42)
    
    best_global_score = 0
    best_global_result = None
    
    # Try multiple aspect ratios
    # Perimeter 4 => 2(w+h)=4 => w+h=2
    # Aspect ratio r = w/h. w = 2r/(1+r), h = 2/(1+r)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 1.1, 0.9]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline, num_trials=8)
        
        if pos is None:
            continue
        
        if score > best_global_score:
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
