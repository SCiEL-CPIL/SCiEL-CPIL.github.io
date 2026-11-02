"""Circle packing in perimeter-4 rectangle: Smooth Log-Sum-Exp Optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _smooth_objective(z, beta=200.0, penalty_weight=1.0):
    """
    Objective function using log-sum-exp smooth penalties for constraints.
    Minimizes -sum(r) + penalty * (smooth_overlap + smooth_boundary).
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    # Hard bounds check to avoid NaNs in log/exp
    if w < 0.1 or h < 0.1 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # 1. Pairwise Overlap Penalty
    # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
    # Constraint: d_ij >= r_i + r_j
    # Violation: v_ij = r_i + r_j - d_ij
    # Smooth penalty: log(1 + exp(beta * v_ij))
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    violations = r_sum_pairs - d_pairs
    
    # Stable log-sum-exp calculation
    # log(1 + exp(beta * v))
    # For large positive v, this is ~ beta * v + log(2)
    # For large negative v, this is ~ 0
    # We use a stable formulation to avoid overflow
    max_v = np.max(violations)
    if max_v > 0:
        # log(1 + exp(beta * v)) = beta * max_v + log(exp(-beta * max_v) + exp(beta * (v - max_v)))
        # Actually simpler: logaddexp(0, beta * v)
        smooth_pen_pairs = np.logaddexp(0, beta * violations)
    else:
        # If all violations are negative, exp(beta * v) is small
        # log(1 + exp(x)) ~ exp(x) for x << 0
        smooth_pen_pairs = np.exp(beta * violations)
        
    obj += penalty_weight * np.sum(smooth_pen_pairs)

    # 2. Boundary Penalty
    # Constraints:
    # x_i >= r_i  => v = r_i - x_i
    # x_i <= w - r_i => v = x_i + r_i - w
    # y_i >= r_i  => v = r_i - y_i
    # y_i <= h - r_i => v = y_i + r_i - h
    
    v_left = r - x
    v_right = x + r - w
    v_bottom = r - y
    v_top = y + r - h
    
    all_violations = np.concatenate([v_left, v_right, v_bottom, v_top])
    
    max_v_bd = np.max(all_violations)
    if max_v_bd > 0:
        smooth_pen_bd = np.logaddexp(0, beta * all_violations)
    else:
        smooth_pen_bd = np.exp(beta * all_violations)
        
    obj += penalty_weight * np.sum(smooth_pen_bd)

    # 3. Aspect Ratio Penalty (keep w in reasonable range [0.5, 1.5])
    # This helps prevent degenerate rectangles
    if w < 0.5:
        obj += 10.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 10.0 * (w - 1.5) ** 2

    return obj


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0 or h <= 0:
        return False
    if np.any(r < 0):
        return False
    
    # Boundary checks
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
        
    # Pairwise overlap checks
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    # Check if d^2 >= (r_i + r_j)^2
    # Using squared distances to avoid sqrt
    if np.any(d_sq_pairs < r_sum_pairs ** 2 - tol):
        return False
        
    return True


def _polish_radii(x, y, w, h, time_limit=2.0):
    """
    Given fixed positions x, y and rectangle w, h, maximize sum(r)
    subject to non-overlap and boundary constraints.
    This is a convex optimization problem (SOCP).
    We can solve it using SLSQP with only radius variables, or simpler heuristics.
    Here we use a simple iterative growth or a tight SLSQP on radii only.
    """
    start_time = time.time()
    r = np.full(N, 0.01) # Small initial radius
    
    # Initial guess based on distance to walls
    for i in range(N):
        r[i] = min(x[i], w - x[i], y[i], h - y[i]) * 0.8
        
    def obj_radii(r_vec):
        # Minimize -sum(r) + penalty for overlaps
        # We use a soft penalty here to make it smooth for SLSQP
        x = x
        y = y
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_vec[:, None] + r_vec[None, :]
        
        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        violations = r_sum_pairs - d_pairs
        
        # Soft penalty
        beta = 50.0
        pen = np.sum(np.logaddexp(0, beta * violations))
        
        # Boundary violations
        v_left = r_vec - x
        v_right = x + r_vec - w
        v_bottom = r_vec - y
        v_top = y + r_vec - h
        all_v = np.concatenate([v_left, v_right, v_bottom, v_top])
        pen += np.sum(np.logaddexp(0, beta * all_v))
        
        return -np.sum(r_vec) + 1.0 * pen

    bounds = [(0.001, 1.0)] * N
    
    try:
        res = minimize(obj_radii, r, method='SLSQP', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-10})
        if res.success:
            r_opt = res.x
            if _is_valid(x, y, r_opt, w, h):
                return r_opt
    except Exception:
        pass
        
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Multi-start configurations
    # Aspect ratios w (width), h = 2 - w
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9]
    
    # Initial layout types
    # 0: Grid, 1: Hexagonal, 2: Random
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 10:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
            
        for layout_type in [0, 1, 2]:
            if time.time() > deadline - 5:
                break
                
            z0 = np.zeros(3 * N + 1)
            
            if layout_type == 0: # Grid
                side = int(np.ceil(np.sqrt(N)))
                for i in range(N):
                    row, col = divmod(i, side)
                    cx = (col + 0.5) / side * w0
                    cy = (row + 0.5) / side * h0
                    r0 = min(w0 / side, h0 / side) * 0.45
                    z0[3*i] = cx
                    z0[3*i+1] = cy
                    z0[3*i+2] = r0
            elif layout_type == 1: # Hexagonal
                rows = 5
                cols = 5
                spacing_x = w0 / (cols + 0.5)
                spacing_y = h0 / (rows + 0.5)
                for i in range(N):
                    row, col = divmod(i, cols)
                    cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
                    cy = (row + 0.5) * spacing_y
                    r0 = min(spacing_x, spacing_y) * 0.48
                    cx = np.clip(cx, r0, w0 - r0)
                    cy = np.clip(cy, r0, h0 - r0)
                    z0[3*i] = cx
                    z0[3*i+1] = cy
                    z0[3*i+2] = r0
            else: # Random
                rng = np.random.default_rng(42 + int(w0*100) + layout_type)
                for i in range(N):
                    r0 = rng.uniform(0.05, 0.15)
                    cx = rng.uniform(r0, w0 - r0)
                    cy = rng.uniform(r0, h0 - r0)
                    z0[3*i] = cx
                    z0[3*i+1] = cy
                    z0[3*i+2] = r0
                    
            z0[-1] = w0
            
            # Optimize
            # Use smooth objective
            # Increase beta and penalty weight gradually? Or just use fixed high values.
            # High beta makes it stiff. Let's use beta=100, penalty=1.0
            # SLSQP might struggle with stiff penalties. 
            # Alternative: Use L-BFGS-B with the smooth penalty. It's faster per iteration.
            
            try:
                # First pass: L-BFGS-B for speed
                res = minimize(_smooth_objective, z0, method='L-BFGS-B',
                               bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                               args=(100.0, 1.0),
                               options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 50})
                
                z_opt = res.x
                
                # Second pass: SLSQP for precision
                # Use a lower beta for smoother landscape in final polish?
                # Or keep same.
                res2 = minimize(_smooth_objective, z_opt, method='SLSQP',
                                bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                                args=(200.0, 1.0),
                                options={'maxiter': 100, 'ftol': 1e-14, 'maxls': 80})
                
                z_opt = res2.x
                
                x = z_opt[0::3][:N]
                y = z_opt[1::3][:N]
                r = z_opt[2::3][:N]
                w = z_opt[-1]
                h = 2.0 - w
                
                # Polish radii given positions
                r_polished = _polish_radii(x, y, w, h, time_limit=1.0)
                
                # Check validity
                if _is_valid(x, y, r_polished, w, h):
                    val = np.sum(r_polished)
                    if val > best_value:
                        best_value = val
                        best_result = np.empty(3*N+1)
                        best_result[0::3][:N] = x
                        best_result[1::3][:N] = y
                        best_result[2::3][:N] = r_polished
                        best_result[-1] = w
                        
            except Exception:
                continue
                
    # If we have a best result, do a final global polish
    if best_result is not None:
        # Try to optimize the whole thing again with the best result as start
        try:
            res_final = minimize(_smooth_objective, best_result, method='SLSQP',
                                 bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                                 args=(300.0, 1.0),
                                 options={'maxiter': 100, 'ftol': 1e-15, 'maxls': 100})
            z_final = res_final.x
            x = z_final[0::3][:N]
            y = z_final[1::3][:N]
            r = z_final[2::3][:N]
            w = z_final[-1]
            h = 2.0 - w
            
            r_polished = _polish_radii(x, y, w, h, time_limit=1.0)
            
            if _is_valid(x, y, r_polished, w, h):
                val = np.sum(r_polished)
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r_polished
                    best_result[-1] = w
        except Exception:
            pass

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    
    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        w = best_result[-1]
        h = 2.0 - w
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
