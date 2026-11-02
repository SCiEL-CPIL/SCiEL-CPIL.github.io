"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _sigmoid(x):
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def _make_grid(w, h, rows, cols, n):
    """Generate a grid initialization for n circles."""
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        xs[i] = (col + 0.5) / cols * w
        ys[i] = (row + 0.5) / rows * h
    return xs, ys

def _make_hex(w, h, n):
    """Generate a hexagonal-like initialization."""
    xs = np.zeros(n)
    ys = np.zeros(n)
    # Try different row counts
    best = None
    for rows in range(1, n + 1):
        cols = (n + rows - 1) // rows
        xs_tmp = np.zeros(n)
        ys_tmp = np.zeros(n)
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols * w
            y = (row + 0.5) / rows * h
            if row % 2 == 1:
                x += 0.5 / cols * w
            xs_tmp[i] = x
            ys_tmp[i] = y
        # Check if all within bounds
        valid = True
        for i in range(n):
            if xs_tmp[i] < 0 or xs_tmp[i] > w or ys_tmp[i] < 0 or ys_tmp[i] > h:
                valid = False
                break
        if valid:
            if best is None or rows < best[0]:
                best = (rows, xs_tmp.copy(), ys_tmp.copy())
    if best is None:
        return _make_grid(w, h, 5, 5, n)
    return best[1], best[2]

def _make_random(w, h, n, seed=42):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1*w, 0.9*w, n)
    ys = rng.uniform(0.1*h, 0.9*h, n)
    return xs, ys

def _objective_equal_radius(params, w, h, W=100.0):
    """
    Objective for equal-radius packing: minimize -r + penalty.
    params = [x1, y1, x2, y2, ..., r] (2*N + 1)
    """
    n = N
    xs = params[0:2*n:2]
    ys = params[1:2*n:2]
    r = params[-1]
    if r < 1e-8:
        r = 1e-8
    
    obj = -r
    grad = np.zeros_like(params)
    grad[-1] = -1.0
    
    # Boundary penalties
    for i in range(n):
        xi, yi = xs[i], ys[i]
        # Left: xi - r < 0
        v = r - xi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[2*i] -= W * s
            grad[-1] += W * s
        # Right: xi + r > w
        v = xi + r - w
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[2*i] += W * s
            grad[-1] += W * s
        # Bottom: yi - r < 0
        v = r - yi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[2*i+1] -= W * s
            grad[-1] += W * s
        # Top: yi + r > h
        v = yi + r - h
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[2*i+1] += W * s
            grad[-1] += W * s
    
    # Pairwise overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_d = 2.0 * r
            if dist_sq < min_d * min_d:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                v = min_d - dist
                if v > -20:
                    s = _sigmoid(v)
                    obj += W * _softplus(v)
                    coeff = W * s / dist
                    grad[2*i] -= coeff * dx
                    grad[2*i+1] -= coeff * dy
                    grad[2*j] += coeff * dx
                    grad[2*j+1] += coeff * dy
                    grad[-1] += 2.0 * W * s
    
    return obj, grad

def _objective_variable_radius(params, w, h, W=100.0):
    """
    Objective for variable-radius packing: minimize -sum(ri) + penalties.
    params = [x1, y1, r1, x2, y2, r2, ...] (3*N)
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    obj = -np.sum(rs)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    
    # Boundary penalties
    for i in range(n):
        xi, yi, ri = xs[i], ys[i], rs[i]
        if ri < 1e-8:
            ri = 1e-8
        v = ri - xi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i] -= W * s
            grad[3*i+2] += W * s
        v = xi + ri - w
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i] += W * s
            grad[3*i+2] += W * s
        v = ri - yi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i+1] -= W * s
            grad[3*i+2] += W * s
        v = yi + ri - h
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i+1] += W * s
            grad[3*i+2] += W * s
    
    # Pairwise overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_d = rs[i] + rs[j]
            if dist_sq < min_d * min_d:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                v = min_d - dist
                if v > -20:
                    s = _sigmoid(v)
                    obj += W * _softplus(v)
                    coeff = W * s / dist
                    grad[3*i] -= coeff * dx
                    grad[3*i+1] -= coeff * dy
                    grad[3*i+2] += W * s
                    grad[3*j] += coeff * dx
                    grad[3*j+1] += coeff * dy
                    grad[3*j+2] += W * s
    
    return obj, grad

def _check_valid_equal(xs, ys, r, w, h, tol=1e-7):
    """Check validity of equal-radius configuration."""
    if r < 1e-6:
        return False
    for i in range(N):
        if xs[i] - r < -tol or xs[i] + r > w + tol:
            return False
        if ys[i] - r < -tol or ys[i] + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if d < 2*r - tol:
                return False
    return True

def _check_valid_var(xs, ys, rs, w, h, tol=1e-7):
    """Check validity of variable-radius configuration."""
    for i in range(N):
        if rs[i] < 1e-6:
            return False
        if xs[i] - rs[i] < -tol or xs[i] + rs[i] > w + tol:
            return False
        if ys[i] - rs[i] < -tol or ys[i] + rs[i] > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True

def _bbox_perim(xs, ys, rs):
    """Compute bounding box perimeter."""
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    return 2.0 * ((max_x - min_x) + (max_y - min_y))

def _scale_to_perim(xs, ys, rs, target_perim=4.0):
    """Scale configuration to target perimeter."""
    perim = _bbox_perim(xs, ys, rs)
    if perim <= target_perim + 1e-9:
        return xs, ys, rs, perim
    scale = target_perim / perim
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs_new = cx + (xs - cx) * scale
    ys_new = cy + (ys - cy) * scale
    rs_new = rs * scale
    return xs_new, ys_new, rs_new, target_perim

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_value = -np.inf
    best_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # --- Equal radius approach ---
        # Generate several initializations
        inits = []
        
        # Grid
        for rows, cols in [(3, 7), (4, 6), (5, 5), (7, 3), (6, 4)]:
            if rows * cols >= N:
                xs, ys = _make_grid(w, h, rows, cols, N)
                inits.append((xs.copy(), ys.copy()))
        
        # Hex
        xs, ys = _make_hex(w, h, N)
        inits.append((xs.copy(), ys.copy()))
        
        # Random
        for seed in [1, 2, 3, 42, 100]:
            xs, ys = _make_random(w, h, N, seed)
            inits.append((xs.copy(), ys.copy()))
        
        for xs_init, ys_init in inits:
            if time.time() > deadline - 8:
                break
            
            # Estimate initial radius
            r_est = 0.0
            for i in range(N):
                min_dist = min(xs_init[i], w - xs_init[i], ys_init[i], h - ys_init[i])
                for j in range(i+1, N):
                    d = np.sqrt((xs_init[i]-xs_init[j])**2 + (ys_init[i]-ys_init[j])**2)
                    min_dist = min(min_dist, d/2.0)
                r_est = min(r_est if r_est > 0 else 1e9, min_dist)
            if r_est <= 0:
                r_est = 0.05 * min(w, h)
            
            params_eq = np.concatenate([np.column_stack([xs_init, ys_init]).flatten(), [r_est]])
            
            bounds_eq = [(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)]
            
            # SLSQP optimization
            try:
                res = minimize(
                    lambda p: _objective_equal_radius(p, w, h),
                    params_eq,
                    jac=True,
                    method='SLSQP',
                    bounds=bounds_eq,
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                p_opt = res.x
                xs_opt = p_opt[0:2*N:2]
                ys_opt = p_opt[1:2*N:2]
                r_opt = p_opt[-1]
                
                if _check_valid_equal(xs_opt, ys_opt, r_opt, w, h):
                    val = N * r_opt
                    if val > best_value:
                        best_value = val
                        best_result = np.column_stack([xs_opt, ys_opt, np.full(N, r_opt)])
            except Exception:
                pass
            
            # L-BFGS-B with bounds
            if time.time() < deadline - 5:
                try:
                    res2 = minimize(
                        lambda p: _objective_equal_radius(p, w, h),
                        params_eq,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds_eq,
                        options={'maxiter': 300, 'ftol': 1e-12}
                    )
                    p_opt2 = res2.x
                    xs_opt2 = p_opt2[0:2*N:2]
                    ys_opt2 = p_opt2[1:2*N:2]
                    r_opt2 = p_opt2[-1]
                    
                    if _check_valid_equal(xs_opt2, ys_opt2, r_opt2, w, h):
                        val2 = N * r_opt2
                        if val2 > best_value:
                            best_value = val2
                            best_result = np.column_stack([xs_opt2, ys_opt2, np.full(N, r_opt2)])
                except Exception:
                    pass
        
        # --- Variable radius approach ---
        # Use grid initialization
        xs_g, ys_g = _make_grid(w, h, 5, 5, N)
        r_init = 0.08 * min(w, h)
        params_var = np.concatenate([xs_g, ys_g, np.full(N, r_init)])
        
        bounds_var = [(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N
        
        try:
            res_v = minimize(
                lambda p: _objective_variable_radius(p, w, h),
                params_var,
                jac=True,
                method='SLSQP',
                bounds=bounds_var,
                options={'maxiter': 400, 'ftol': 1e-12}
            )
            p_v = res_v.x
            xs_v = p_v[0::3]
            ys_v = p_v[1::3]
            rs_v = p_v[2::3]
            
            if _check_valid_var(xs_v, ys_v, rs_v, w, h):
                val_v = np.sum(rs_v)
                if val_v > best_value:
                    best_value = val_v
                    best_result = np.column_stack([xs_v, ys_v, rs_v])
        except Exception:
            pass
    
    # Final refinement: try to improve best_result with local search
    if best_result is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            perturbed = best_result + np.random.randn(*best_result.shape) * 0.005
            # Clip to reasonable bounds
            w_cur = 2.0 * best_result[:, 0].max() / (best_result[:, 0].max() + best_result[:, 1].max()) if best_result[:, 1].max() > 0 else 1.0
            h_cur = 2.0 - w_cur
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 2.0)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 2.0)
            perturbed[:, 2] = np.clip(perturbed[:, 2], 1e-4, 0.5)
            
            # Quick validity check and value
            if _check_valid_var(perturbed[:, 0], perturbed[:, 1], perturbed[:, 2], 1.0, 1.0):
                # Re-optimize from perturbed
                pass  # Skip for now to save time
    
    # Return result
    if best_result is None:
        # Fallback: simple grid
        xs, ys = _make_grid(1.0, 1.0, 5, 5, N)
        r = 0.08
        best_result = np.column_stack([xs, ys, np.full(N, r)])
    
    # Scale to ensure perimeter <= 4
    xs_f = best_result[:, 0]
    ys_f = best_result[:, 1]
    rs_f = best_result[:, 2]
    
    perim = _bbox_perim(xs_f, ys_f, rs_f)
    if perim > 4.0 + 1e-9:
        scale = 4.0 / perim
        cx = np.mean(xs_f)
        cy = np.mean(ys_f)
        xs_f = cx + (xs_f - cx) * scale
        ys_f = cy + (ys_f - cy) * scale
        rs_f = rs_f * scale
    
    return np.column_stack([xs_f, ys_f, rs_f])
