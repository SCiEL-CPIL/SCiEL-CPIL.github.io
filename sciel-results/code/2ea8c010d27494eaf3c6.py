"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate_and_compute(radii, x, y, W, H):
    """Check validity and compute sum of radii. Returns (is_valid, value)."""
    n = len(radii)
    # Check bounds
    for i in range(n):
        if x[i] < radii[i] - 1e-9 or x[i] > W - radii[i] + 1e-9:
            return False, 0.0
        if y[i] < radii[i] - 1e-9 or y[i] > H - radii[i] + 1e-9:
            return False, 0.0
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist*min_dist - 1e-9:
                return False, 0.0
    # Check perimeter
    perim = 2*(W + H)
    if perim > 4.0 + 1e-9:
        return False, 0.0
    value = np.sum(radii)
    return True, value


def _smooth_penalty(radii, x, y, W, H, beta=100.0):
    """Compute smooth penalty for violations using soft-min approximation."""
    n = len(radii)
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        # Left boundary violation
        pen = max(0, radii[i] - x[i])
        penalty += pen * pen * beta
        # Right boundary violation
        pen = max(0, x[i] + radii[i] - W)
        penalty += pen * pen * beta
        # Bottom boundary violation
        pen = max(0, radii[i] - y[i])
        penalty += pen * pen * beta
        # Top boundary violation
        pen = max(0, y[i] + radii[i] - H)
        penalty += pen * pen * beta
    
    # Pairwise distance violations
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            pen = max(0, min_dist - dist)
            penalty += pen * pen * beta
    
    # Perimeter violation
    perim = 2*(W + H)
    pen = max(0, perim - 4.0)
    penalty += pen * pen * beta * 10
    
    return penalty


def _objective(params, W, H, beta=100.0):
    """Objective function: minimize negative sum of radii + penalty."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    radii = params[2*n:]
    
    # Radii must be positive
    radii = np.maximum(radii, 0.01)
    
    penalty = _smooth_penalty(radii, x, y, W, H, beta)
    value = np.sum(radii)
    
    return penalty - value


def _seed_grid():
    """Create a 5x5 grid of circles in a 1x1 square (perimeter 4)."""
    side = 5
    W = 1.0
    H = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:N])


def _optimize_scipy(initial, W, H, deadline, beta_start=50.0, beta_end=500.0):
    """Optimize using scipy with smooth penalty, increasing beta over time."""
    n = N
    x0 = initial[:, 0].copy()
    y0 = initial[:, 1].copy()
    r0 = initial[:, 2].copy()
    
    params0 = np.concatenate([x0, y0, r0])
    
    # Bounds: x in [r, W-r], y in [r, H-r], r in [0.01, min(W,H)/2]
    # We'll use loose bounds and let penalty handle tight constraints
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))  # x
    for i in range(n):
        bounds.append((0.0, H))  # y
    for i in range(n):
        bounds.append((0.01, min(W, H)/2))  # r
    
    best_params = params0.copy()
    best_val = 0.0
    _, best_val = _validate_and_compute(r0, x0, y0, W, H)
    
    # Anneal beta from low to high
    n_iterations = 5
    for iter in range(n_iterations):
        if time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (iter / max(1, n_iterations-1))
        
        # Try different initial points
        for restart in range(3):
            if time.time() > deadline:
                break
            
            if restart == 0:
                x_init = x0.copy()
                y_init = y0.copy()
                r_init = r0.copy()
            else:
                x_init = x0 + np.random.uniform(-0.05, 0.05, n)
                y_init = y0 + np.random.uniform(-0.05, 0.05, n)
                r_init = np.maximum(r0 + np.random.uniform(-0.02, 0.02, n), 0.01)
            
            params_init = np.concatenate([x_init, y_init, r_init])
            
            try:
                result = minimize(
                    _objective,
                    params_init,
                    args=(W, H, beta),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 0:  # Negative fun means value > penalty
                    params = result.x
                    x_new = params[:n]
                    y_new = params[n:2*n]
                    r_new = np.maximum(params[2*n:], 0.01)
                    
                    # Clamp to valid range
                    for i in range(n):
                        r_new[i] = min(r_new[i], W/2, H/2)
                        x_new[i] = np.clip(x_new[i], r_new[i], W - r_new[i])
                        y_new[i] = np.clip(y_new[i], r_new[i], H - r_new[i])
                    
                    is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
                    if is_valid and val > best_val:
                        best_val = val
                        best_params = np.concatenate([x_new, y_new, r_new])
                        x0 = x_new.copy()
                        y0 = y_new.copy()
                        r0 = r_new.copy()
                        
            except Exception:
                pass
    
    x_best = best_params[:n]
    y_best = best_params[n:2*n]
    r_best = np.maximum(best_params[2*n:], 0.01)
    
    return np.column_stack([x_best, y_best, r_best]), best_val


def _local_refine(circles, W, H, deadline, max_iter=500):
    """Fine-tune with local search on individual circles."""
    n = len(circles)
    best_x = circles[:, 0].copy()
    best_y = circles[:, 1].copy()
    best_r = circles[:, 2].copy()
    _, best_val = _validate_and_compute(best_r, best_x, best_y, W, H)
    
    step_size = 0.02
    for iter in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = np.random.randint(n)
        x_new = best_x.copy()
        y_new = best_y.copy()
        r_new = best_r.copy()
        
        x_new[idx] += np.random.uniform(-step_size, step_size)
        y_new[idx] += np.random.uniform(-step_size, step_size)
        r_new[idx] = max(0.01, r_new[idx] + np.random.uniform(-0.005, 0.005))
        
        # Clamp
        r_new[idx] = min(r_new[idx], W/2, H/2)
        x_new[idx] = np.clip(x_new[idx], r_new[idx], W - r_new[idx])
        y_new[idx] = np.clip(y_new[idx], r_new[idx], H - r_new[idx])
        
        is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
        if is_valid and val > best_val:
            best_x = x_new
            best_y = y_new
            best_r = r_new
            best_val = val
            step_size *= 0.99
        else:
            step_size = min(0.05, step_size * 1.01)
    
    return np.column_stack([best_x, best_y, best_r]), best_val


def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with grid
    initial = _seed_grid()
    
    # Try different rectangle aspect ratios
    best_result = initial.copy()
    best_val = 0.0
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W + H = 2, so H = 2 - W
        # ar = W/H, so W = ar * H = ar * (2 - W) => W(1+ar) = 2*ar => W = 2*ar/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        if W <= 0 or H <= 0:
            continue
        
        # Scale initial grid to fit this rectangle
        scale_x = W / 1.0
        scale_y = H / 1.0
        scaled = initial.copy()
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        scaled[:, 2] *= min(scale_x, scale_y)  # Scale radii by min factor
        
        # Optimize with scipy
        result, val = _optimize_scipy(scaled, W, H, deadline)
        
        # Local refinement
        if val > 0:
            result, val = _local_refine(result, W, H, deadline)
        
        if val > best_val:
            best_val = val
            best_result = result
    
    # Final validation
    x = best_result[:, 0]
    y = best_result[:, 1]
    r = best_result[:, 2]
    
    # Find actual bounding box
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W_actual = x_max - x_min
    H_actual = y_max - y_min
    
    # If perimeter exceeds 4, we need to scale down
    perim = 2 * (W_actual + H_actual)
    if perim > 4.0:
        scale = 4.0 / perim
        r = r * scale
        x = x_min + (x - x_min) * scale
        y = y_min + (y - y_min) * scale
        best_result = np.column_stack([x, y, r])
    
    return best_result
# EVOLVE-BLOCK-END
