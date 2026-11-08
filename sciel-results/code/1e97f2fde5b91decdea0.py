"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate_and_compute(radii, x, y, W, H):
    """Check validity and compute sum of radii. Returns (is_valid, value)."""
    n = len(radii)
    for i in range(n):
        if x[i] < radii[i] - 1e-9 or x[i] > W - radii[i] + 1e-9:
            return False, 0.0
        if y[i] < radii[i] - 1e-9 or y[i] > H - radii[i] + 1e-9:
            return False, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist - 1e-9:
                return False, 0.0
    perim = 2 * (W + H)
    if perim > 4.0 + 1e-9:
        return False, 0.0
    return True, float(np.sum(radii))


def _soft_min_sum(vals, beta=20.0):
    """Smooth approximation of sum of max(0, v) using soft-min."""
    # For each violation v, max(0,v) ≈ v - softplus(-v) where softplus(x)=log(1+exp(x))/beta
    # Actually simpler: max(0,v) ≈ (v + sqrt(v^2 + eps))/2 for smoothness
    # Or use: max(0,v) ≈ v * sigmoid(beta*v)
    s = 0.0
    for v in vals:
        if v > 0:
            s += v * (1.0 / (1.0 + math.exp(-beta * v)))
    return s


def _objective(params, W, H, beta=50.0):
    """Objective: minimize -sum(radii) + penalty."""
    n = N
    x = params[:n]
    y = params[n:2 * n]
    r = params[2 * n:]
    r = np.maximum(r, 0.005)
    
    # Boundary violations
    viol = []
    for i in range(n):
        viol.append(r[i] - x[i])
        viol.append(x[i] + r[i] - W)
        viol.append(r[i] - y[i])
        viol.append(y[i] + r[i] - H)
    
    # Pairwise violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx * dx + dy * dy)
            viol.append(r[i] + r[j] - dist)
    
    # Perimeter violation
    perim = 2 * (W + H)
    viol.append(perim - 4.0)
    
    # Smooth penalty
    pen = _soft_min_sum(viol, beta)
    
    return pen - np.sum(r)


def _seed_hex(W, H):
    """Create hexagonal packing seed."""
    # Try to fit a hex grid
    circles = []
    # Start with a reasonable radius estimate
    # Area of rectangle = W*H, area of 21 circles = 21*pi*r^2
    # r ≈ sqrt(W*H / (21*pi * packing_fraction))
    packing_frac = 0.85  # hex packing is ~0.907
    r_est = math.sqrt(W * H / (21 * math.pi * packing_frac))
    
    # Place circles in a grid pattern first, then optimize
    # Use a 5x5 grid with hexagonal offset
    side = 5
    for i in range(side):
        for j in range(side):
            if len(circles) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            # Hexagonal offset for odd rows
            if j % 2 == 1:
                cx += W / (2 * side)
            cx = min(max(cx, 0.1), W - 0.1)
            cy = min(max(cy, 0.1), H - 0.1)
            circles.append([cx, cy, r_est * 0.9])
    
    return np.array(circles[:N])


def _seed_grid(W, H):
    """Create a simple grid seed."""
    side = 5
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) * 0.95
            circles.append([cx, cy, r])
    return np.array(circles[:N])


def _optimize_scipy(circles_init, W, H, deadline, beta_start=20.0, beta_end=500.0):
    """Optimize using scipy L-BFGS-B with annealed penalty."""
    n = N
    x0 = circles_init[:, 0].copy()
    y0 = circles_init[:, 1].copy()
    r0 = circles_init[:, 2].copy()
    
    params0 = np.concatenate([x0, y0, r0])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))
    for i in range(n):
        bounds.append((0.0, H))
    for i in range(n):
        bounds.append((0.005, min(W, H) / 2))
    
    best_params = params0.copy()
    best_val = 0.0
    _, best_val = _validate_and_compute(r0, x0, y0, W, H)
    
    # Anneal beta
    n_iterations = 6
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (it / max(1, n_iterations - 1))
        
        for restart in range(4):
            if time.time() > deadline:
                break
            
            if restart == 0:
                p_init = params0.copy()
            else:
                p_init = params0 + np.random.normal(0, 0.03 * (1 + 0.5*restart), len(params0))
                p_init = np.clip(p_init, 0, None)
            
            try:
                result = minimize(
                    _objective,
                    p_init,
                    args=(W, H, beta),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                params = result.x
                x_new = np.clip(params[:n], 0, W)
                y_new = np.clip(params[n:2*n], 0, H)
                r_new = np.maximum(params[2*n:], 0.005)
                
                # Clamp to valid
                for i in range(n):
                    r_new[i] = min(r_new[i], W / 2, H / 2)
                    x_new[i] = np.clip(x_new[i], r_new[i], W - r_new[i])
                    y_new[i] = np.clip(y_new[i], r_new[i], H - r_new[i])
                
                is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
                if is_valid and val > best_val:
                    best_val = val
                    best_params = np.concatenate([x_new, y_new, r_new])
                    params0 = best_params.copy()
                    
            except Exception:
                pass
    
    x_best = best_params[:n]
    y_best = best_params[n:2 * n]
    r_best = np.maximum(best_params[2 * n:], 0.005)
    return np.column_stack([x_best, y_best, r_best]), best_val


def _local_refine(circles, W, H, deadline, max_iter=2000):
    """Local search refinement."""
    n = len(circles)
    best_x = circles[:, 0].copy()
    best_y = circles[:, 1].copy()
    best_r = circles[:, 2].copy()
    _, best_val = _validate_and_compute(best_r, best_x, best_y, W, H)
    
    step_size = 0.015
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = np.random.randint(n)
        x_new = best_x.copy()
        y_new = best_y.copy()
        r_new = best_r.copy()
        
        x_new[idx] += np.random.uniform(-step_size, step_size)
        y_new[idx] += np.random.uniform(-step_size, step_size)
        r_new[idx] = max(0.005, r_new[idx] + np.random.uniform(-0.004, 0.004))
        
        r_new[idx] = min(r_new[idx], W / 2, H / 2)
        x_new[idx] = np.clip(x_new[idx], r_new[idx], W - r_new[idx])
        y_new[idx] = np.clip(y_new[idx], r_new[idx], H - r_new[idx])
        
        is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
        if is_valid and val > best_val:
            best_x = x_new
            best_y = y_new
            best_r = r_new
            best_val = val
            step_size *= 0.995
        else:
            step_size = min(0.04, step_size * 1.005)
    
    return np.column_stack([best_x, best_y, best_r]), best_val


def _polish(circles, W, H, deadline):
    """High-precision polish with tighter tolerances."""
    n = len(circles)
    x0 = circles[:, 0].copy()
    y0 = circles[:, 1].copy()
    r0 = circles[:, 2].copy()
    params0 = np.concatenate([x0, y0, r0])
    
    bounds = []
    for i in range(n):
        bounds.append((max(0, r0[i] - 0.01), W - max(0, r0[i] - 0.01)))
    for i in range(n):
        bounds.append((max(0, r0[i] - 0.01), H - max(0, r0[i] - 0.01)))
    for i in range(n):
        bounds.append((max(0.005, r0[i] - 0.005), min(W, H) / 2))
    
    best_params = params0.copy()
    _, best_val = _validate_and_compute(r0, x0, y0, W, H)
    
    # Use very high beta for tight constraints
    for beta in [200.0, 500.0, 1000.0]:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _objective,
                best_params,
                args=(W, H, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            params = result.x
            x_new = np.clip(params[:n], 0, W)
            y_new = np.clip(params[n:2*n], 0, H)
            r_new = np.maximum(params[2*n:], 0.005)
            for i in range(n):
                r_new[i] = min(r_new[i], W / 2, H / 2)
                x_new[i] = np.clip(x_new[i], r_new[i], W - r_new[i])
                y_new[i] = np.clip(y_new[i], r_new[i], H - r_new[i])
            is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
            if is_valid and val > best_val:
                best_val = val
                best_params = np.concatenate([x_new, y_new, r_new])
        except Exception:
            pass
    
    x_best = best_params[:n]
    y_best = best_params[n:2 * n]
    r_best = np.maximum(best_params[2 * n:], 0.005)
    return np.column_stack([x_best, y_best, r_best]), best_val


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_result = None
    best_val = 0.0
    
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.3, 0.65, 1.5, 0.55]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        if W <= 0.1 or H <= 0.1:
            continue
        
        # Try multiple seeds
        seeds = [_seed_grid(W, H), _seed_hex(W, H)]
        
        for seed in seeds:
            if time.time() > deadline:
                break
            
            # Scale seed radii to be reasonable
            scale = min(W, H) / 2.0
            seed[:, 2] = np.clip(seed[:, 2], 0.01, scale * 0.4)
            
            # Scipy optimization
            result, val = _optimize_scipy(seed, W, H, deadline)
            
            # Local refinement
            if val > 0:
                result, val = _local_refine(result, W, H, deadline)
            
            # Polish
            if val > 0 and time.time() < deadline - 20:
                result, val = _polish(result, W, H, deadline)
            
            if val > best_val:
                best_val = val
                best_result = result
    
    if best_result is None:
        # Fallback: simple grid in 1x1
        W = 1.0
        H = 1.0
        best_result = _seed_grid(W, H)
        best_val = float(np.sum(best_result[:, 2]))
    
    # Final validation and scaling
    x = best_result[:, 0]
    y = best_result[:, 1]
    r = best_result[:, 2]
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W_actual = x_max - x_min
    H_actual = y_max - y_min
    
    perim = 2 * (W_actual + H_actual)
    if perim > 4.0:
        scale = 4.0 / perim
        r = r * scale
        x = x_min + (x - x_min) * scale
        y = y_min + (y - y_min) * scale
        best_result = np.column_stack([x, y, r])
    
    return best_result

# EVOLVE-BLOCK-END
