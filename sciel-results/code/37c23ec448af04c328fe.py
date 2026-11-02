"""Circle packing in perimeter-4 rectangle: SOCP-inspired ADMM + SLSQP refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _check_validity(arr, tol=1e-8):
    """Check if circles are valid: disjoint interiors, inside bounding box, perimeter <= 4."""
    n = arr.shape[0]
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 0):
        return False
    
    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.max()
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter constraint: 2*(w + h) <= 4 => w + h <= 2
    if w + h > 2.0 + tol:
        return False
    
    # Circles must be inside the bounding box
    for i in range(n):
        if xs[i] - rs[i] < x_min - tol or xs[i] + rs[i] > x_max + tol:
            return False
        if ys[i] - rs[i] < y_min - tol or ys[i] + rs[i] > y_max + tol:
            return False
    
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            r_sum = rs[i] + rs[j]
            if d2 < r_sum * r_sum - tol:
                return False
    
    return True


def _compute_value(arr):
    """Compute the sum of radii."""
    return float(np.sum(arr[:, 2]))


def _vectorized_objective(z, penalty_weight=500.0):
    """Vectorized objective with smooth penalties for SLSQP."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0.001):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _admm_packing(x, y, r, w, h, rho=10.0, max_iters=200, tol=1e-6):
    """
    ADMM solver for the SOCP formulation of circle packing.
    
    Variables:
    - x, y: center positions
    - r: radii
    - w: width (h = 2 - w)
    
    Constraints (as SOCP):
    - ||(x_i - x_j, y_i - y_j)|| <= r_i + r_j  (non-overlap)
    - x_i >= r_i, x_i <= w - r_i  (boundary x)
    - y_i >= r_i, y_i <= h - r_i  (boundary y)
    
    We use ADMM with augmented Lagrangian.
    """
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    w = float(w)
    h = 2.0 - w
    
    # Dual variables
    # For each pair (i,j): lambda_ij for non-overlap
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    lam_pair = np.zeros(n_pairs)
    
    # For boundary: lambda for each circle and each side
    lam_left = np.zeros(n)
    lam_right = np.zeros(n)
    lam_bottom = np.zeros(n)
    lam_top = np.zeros(n)
    
    prev_x, prev_y, prev_r, prev_w = x.copy(), y.copy(), r.copy(), w
    
    for it in range(max_iters):
        # Update x, y, r, w (minimize augmented Lagrangian)
        # This is a convex QP for fixed duals, but we'll use a simplified approach
        # by doing gradient descent with step size
        
        # Compute gradients
        step = 0.005
        
        # Non-overlap constraints
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-10
            r_sum = r[i] + r[j]
            
            if d < r_sum:
                # Violation: push apart
                overlap = r_sum - d
                gx_i = (dx / d) * (rho * overlap + lam_pair[idx])
                gy_i = (dy / d) * (rho * overlap + lam_pair[idx])
                gx_j = -gx_i
                gy_j = -gy_i
                
                x[i] -= step * gx_i
                y[i] -= step * gy_i
                x[j] -= step * gx_j
                y[j] -= step * gy_j
                
                # Update dual
                lam_pair[idx] += rho * overlap
        
        # Boundary constraints
        for i in range(n):
            # Left: x_i >= r_i
            viol_left = r[i] - x[i]
            if viol_left > 0:
                x[i] += step * (rho * viol_left + lam_left[i])
                lam_left[i] += rho * viol_left
            else:
                lam_left[i] = max(0, lam_left[i])
            
            # Right: x_i <= w - r_i
            viol_right = x[i] + r[i] - w
            if viol_right > 0:
                x[i] -= step * (rho * viol_right + lam_right[i])
                lam_right[i] += rho * viol_right
            else:
                lam_right[i] = max(0, lam_right[i])
            
            # Bottom: y_i >= r_i
            viol_bottom = r[i] - y[i]
            if viol_bottom > 0:
                y[i] += step * (rho * viol_bottom + lam_bottom[i])
                lam_bottom[i] += rho * viol_bottom
            else:
                lam_bottom[i] = max(0, lam_bottom[i])
            
            # Top: y_i <= h - r_i
            viol_top = y[i] + r[i] - h
            if viol_top > 0:
                y[i] -= step * (rho * viol_top + lam_top[i])
                lam_top[i] += rho * viol_top
            else:
                lam_top[i] = max(0, lam_top[i])
        
        # Update radii: increase if possible (gradient ascent on sum of r)
        # But constrained by non-overlap and boundary
        # Simple approach: try to increase each r slightly
        for i in range(n):
            # Check if increasing r_i would violate constraints
            # For simplicity, just do a small increase and let constraints handle it
            r[i] += step * 0.5
            r[i] = max(r[i], 0.01)
        
        # Update w: adjust aspect ratio
        # Try to balance
        avg_x_extent = (x.max() - x.min())
        avg_y_extent = (y.max() - y.min())
        target_w = avg_x_extent + 2 * np.mean(r)
        w += step * 0.1 * (target_w - w)
        w = np.clip(w, 0.3, 1.7)
        h = 2.0 - w
        
        # Check convergence
        diff = np.sqrt(np.sum((x - prev_x)**2) + np.sum((y - prev_y)**2) + 
                       np.sum((r - prev_r)**2) + (w - prev_w)**2)
        prev_x, prev_y, prev_r, prev_w = x.copy(), y.copy(), r.copy(), w
        
        if diff < tol:
            break
        
        # Clamp to valid range
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        r = np.clip(r, 0.01, 0.5)
    
    # Project to valid configuration
    x = np.clip(x, 0, w)
    y = np.clip(y, 0, h)
    r = np.clip(r, 0.01, 0.5)
    
    return x, y, r, w


def _slsqp_refine(x, y, r, w, max_iters=1000, time_limit=10.0):
    """Refine using SLSQP with the vectorized objective."""
    z = np.concatenate([x, y, r, [w]])
    
    def objective(z):
        return _vectorized_objective(z, penalty_weight=500.0)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.5))  # x_i
        bounds.append((0.001, 0.5))  # y_i
        bounds.append((0.001, 0.5))  # r_i
    bounds.append((0.3, 1.7))  # w
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda z: 2.0 - z[-1] - 0.3},  # h >= 0.3
    ]
    
    start_time = time.time()
    
    result = minimize(objective, z, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': max_iters, 
                                                       'ftol': 1e-8})
    
    if result.success or result.fun < objective(z):
        z = result.x
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
    
    return x, y, r, w


def _expand_radii(x, y, r, w, max_iters=50):
    """Try to expand all radii uniformly while maintaining validity."""
    h = 2.0 - w
    best_r = r.copy()
    best_val = np.sum(r)
    
    for it in range(max_iters):
        # Try to scale all radii by a small factor
        scale = 1.0 + 0.005 * (1 - it / max_iters)
        new_r = r * scale
        
        # Check if still valid
        valid = True
        for i in range(N):
            if new_r[i] > x[i] or new_r[i] > w - x[i]:
                valid = False
                break
            if new_r[i] > y[i] or new_r[i] > h - y[i]:
                valid = False
                break
        
        if valid:
            # Check pairwise
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    r_sum = new_r[i] + new_r[j]
                    if d2 < r_sum * r_sum - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            r = new_r
            if np.sum(r) > best_val:
                best_val = np.sum(r)
                best_r = r.copy()
        else:
            break
    
    return best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    def check_time():
        return (time.time() - start_time) > deadline
    
    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        
        # Grid-based initializations with different aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            
            # 5x5 grid
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            for i in range(N):
                row, col = divmod(i, 5)
                x[i] = (col + 0.5) / 5 * w0
                y[i] = (row + 0.5) / 5 * h0
                r[i] = min(w0 / 5, h0 / 5) * 0.45
            
            guesses.append((x, y, r, w0))
        
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = rng.uniform(0.05, w0 - 0.05, N)
            y = rng.uniform(0.05, h0 - 0.05, N)
            r = rng.uniform(0.04, 0.1, N)
            guesses.append((x, y, r, w0))
        
        # Hexagonal pattern
        w0 = 1.0
        h0 = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        spacing = 0.18
        for i in range(N):
            row, col = divmod(i, 5)
            x[i] = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing
            y[i] = (row + 0.5) * spacing * 0.866
            r[i] = spacing * 0.48
        x = np.clip(x, 0.05, w0 - 0.05)
        y = np.clip(y, 0.05, h0 - 0.05)
        guesses.append((x, y, r, w0))
        
        return guesses
    
    guesses = get_initial_guesses()
    
    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        if check_time():
            break
        
        x, y, r, w = x0.copy(), y0.copy(), r0.copy(), w0
        
        # Phase 1: ADMM to get a reasonable feasible solution
        x, y, r, w = _admm_packing(x, y, r, w, rho=10.0, max_iters=100)
        
        # Phase 2: SLSQP refinement
        if not check_time():
            x, y, r, w = _slsqp_refine(x, y, r, w, max_iters=500, time_limit=8.0)
        
        # Phase 3: Expand radii
        if not check_time():
            r = _expand_radii(x, y, r, w, max_iters=30)
        
        # Check validity and compute value
        arr = np.column_stack([x, y, r])
        if _check_validity(arr):
            val = _compute_value(arr)
            if val > best_value:
                best_value = val
                best_result = arr.copy()
    
    # If no valid result, try a simple fallback
    if best_result is None:
        # Simple grid
        w = 1.0
        h = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, 5)
            x[i] = (col + 0.5) / 5 * w
            y[i] = (row + 0.5) / 5 * h
            r[i] = min(w / 5, h / 5) * 0.4
        best_result = np.column_stack([x, y, r])
    
    return best_result
