"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _smooth_objective(params, n, temperature=0.01, penalty_weight=1000.0):
    """
    Smooth objective using log-sum-exp for soft min/max.
    params: flat array of shape (3*n,) containing [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    params = np.asarray(params)
    xy = params[0::3].reshape(-1, 2)
    r = params[1::3].reshape(-1)
    
    r = np.maximum(r, 0.001)
    
    # Wall violations: for each circle, max violation = max(ri-x, ri-(1-x), ri-y, ri-(1-y), 0)
    # Use soft max via log-sum-exp
    wall_violations = np.zeros(n)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        vals = np.array([ri - x, ri - (1.0 - x), ri - y, ri - (1.0 - y), 0.0])
        # Soft max with temperature
        m = np.max(vals)
        wall_violations[i] = m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))
    
    # Overlap violations: for each pair, max(0, ri+rj - dist)
    # Use soft max
    overlap_violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            vals = np.array([min_dist - dist, 0.0])
            m = np.max(vals)
            sv = m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))
            overlap_violations.append(sv)
    
    overlap_violations = np.array(overlap_violations)
    
    penalty = penalty_weight * (np.sum(wall_violations**2) + np.sum(overlap_violations**2))
    total_radius = np.sum(r)
    
    return -total_radius + penalty


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Initial radius: half the cell size minus a small margin
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    # Pad if needed (shouldn't happen for side=ceil(sqrt(N)))
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init():
    """Generate a random valid packing with small circles."""
    positions = []
    r = 0.02
    for i in range(N):
        x = random.uniform(r, 1 - r)
        y = random.uniform(r, 1 - r)
        positions.append([x, y, r])
    return np.array(positions[:N])


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    # Flatten to (M, 2)
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 50
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _gradient_polish(xy, r, deadline, max_iter=300):
    """
    Use L-BFGS-B with smooth penalty terms to polish the packing.
    Returns (xy, r, total, valid) of the best valid packing found.
    """
    n = len(xy)
    
    # Flatten to optimization variable
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    # Bounds: x,y in [0.001, 0.999], r in [0.001, 0.4]
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.4))    # r
    
    best_params = params.copy()
    best_total, best_valid = _eval_packing(xy, r)
    
    # Run L-BFGS-B with smooth objective
    try:
        result = minimize(
            _smooth_objective,
            params,
            args=(n, 0.01, 1000.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        # Extract optimized params
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        # Validate
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    # Try with tighter temperature
    try:
        result2 = minimize(
            _smooth_objective,
            best_params,
            args=(n, 0.005, 2000.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        
        opt_params = result2.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    
    return final_xy, final_r, best_total, best_valid


def _high_precision_polish(xy, r, deadline):
    """
    Final high-precision polish: use smooth objective with very tight tolerances,
    and also try a variable-elimination approach where we optimize positions only
    given fixed radii, then adjust radii.
    """
    n = len(xy)
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    best_total, best_valid = _eval_packing(xy, r)
    best_params = params.copy()
    
    # Run with very tight temperature
    try:
        result = minimize(
            _smooth_objective,
            params,
            args=(n, 0.002, 5000.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-18, 'gtol': 1e-14}
        )
        
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    # Try position-only optimization: fix radii, optimize positions
    try:
        cur_xy = best_params[0::3].reshape(-1, 2)
        cur_r = np.maximum(best_params[1::3], 0.001)
        
        def _pos_only_objective(pos_params):
            """Optimize only positions, radii fixed."""
            pos = pos_params.reshape(-1, 2)
            # Wall violations
            wall_pen = 0.0
            for i in range(n):
                x, y = pos[i, 0], pos[i, 1]
                ri = cur_r[i]
                wall_pen += max(0.0, ri - x)**2
                wall_pen += max(0.0, ri - (1.0 - x))**2
                wall_pen += max(0.0, ri - y)**2
                wall_pen += max(0.0, ri - (1.0 - y))**2
            # Overlap violations
            overlap_pen = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = cur_r[i] + cur_r[j]
                    if dist < min_dist:
                        overlap_pen += (min_dist - dist)**2
            return 1000.0 * (wall_pen + overlap_pen)
        
        pos_params = cur_xy.ravel()
        pos_bounds = [(0.001, 0.999)] * (2 * n)
        
        result_pos = minimize(
            _pos_only_objective,
            pos_params,
            method='L-BFGS-B',
            bounds=pos_bounds,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        
        opt_pos = result_pos.x.reshape(-1, 2)
        total, valid = _eval_packing(opt_pos, cur_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params[0::3] = opt_pos[:, 0]
            best_params[1::3] = opt_pos[:, 1]
            
    except Exception:
        pass
    
    # One more full optimization pass with even tighter settings
    try:
        result2 = minimize(
            _smooth_objective,
            best_params,
            args=(n, 0.001, 10000.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'maxfun': 3000, 'ftol': 1e-20, 'gtol': 1e-15}
        )
        
        opt_params = result2.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    
    return final_xy, final_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Try multiple grid sizes for initialization with a finer search grid
    for side in [5, 6, 7]:
        if time.time() > deadline - 40.0:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        
        # Step 1: Grid-based local search with 80x80 grid
        if time.time() < deadline - 35.0:
            out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 35.0, grid_size=80)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r
    
    # If no valid result found, try random init
    if best_xy is None or not best_valid:
        xy = _random_init()
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        if time.time() < deadline - 35.0:
            out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 35.0, grid_size=80)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r
    
    # If still no valid result, fall back to a simple grid
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        # Quick validation
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            # Last resort: tiny circles in a grid
            side = int(np.ceil(np.sqrt(N)))
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result
    
    # Gradient polish if we have time
    if time.time() < deadline - 15.0:
        g_xy, g_r, g_total, g_valid = _gradient_polish(best_xy, best_r, deadline - 15.0)
        if g_valid and g_total > best_total:
            best_total = g_total
            best_valid = True
            best_xy = g_xy
            best_r = g_r
    
    # Final high-precision polish if we have time
    if time.time() < deadline - 5.0:
        p_xy, p_r, p_total, p_valid = _high_precision_polish(best_xy, best_r, deadline - 3.0)
        if p_valid and p_total > best_total:
            best_total = p_total
            best_valid = True
            best_xy = p_xy
            best_r = p_r
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
