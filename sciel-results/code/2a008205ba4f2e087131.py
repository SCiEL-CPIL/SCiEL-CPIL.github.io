import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit

def _compute_radius(x):
    """Compute the actual radius (score) for a configuration."""
    x = x.reshape(N, 2)
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0)
    min_dist = np.sqrt(np.min(dists_sq))
    
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    
    return min(min_dist / 2.0, min_wall)

def _violation_and_grad(x, r):
    """
    Compute the sum of squared violations of constraints for radius r.
    Constraints:
    1. Pair distance d_ij >= 2r
    2. Wall distance w_i >= r (where w_i = min(0.5 - |x_ix|, 0.5 - |x_iy|))
    
    We use a squared hinge loss: max(0, violation)^2
    """
    x = x.reshape(N, 2)
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Pair violations
    # V_pair_ij = max(0, 2r - d_ij)
    # We only care about pairs where d_ij < 2r
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_violations = np.maximum(0, 2 * r - pair_dists)
    
    # Wall violations
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    # The wall constraint is min(wx, wy) >= r.
    # This is equivalent to wx >= r AND wy >= r.
    # Violation_x = max(0, r - wx)
    # Violation_y = max(0, r - wy)
    wall_viol_x = np.maximum(0, r - wx)
    wall_viol_y = np.maximum(0, r - wy)
    
    # Total objective: sum of squares
    obj_val = np.sum(pair_violations**2) + np.sum(wall_viol_x**2) + np.sum(wall_viol_y**2)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pair gradient
    # d/dx_i (max(0, 2r - d_ij)^2) = 2 * max(0, 2r - d_ij) * (-1) * (x_i - x_j) / d_ij
    # Let v_ij = max(0, 2r - d_ij)
    # If v_ij > 0, grad contribution to x_i is -2 * v_ij * (x_i - x_j) / d_ij
    # grad contribution to x_j is -2 * v_ij * (x_j - x_i) / d_ij = 2 * v_ij * (x_i - x_j) / d_ij
    
    active_pairs = pair_violations > 1e-12
    if np.any(active_pairs):
        active_indices = np.where(active_pairs)[0]
        for idx in active_indices:
            i, j = iu[0][idx], iu[1][idx]
            d = pair_dists[idx]
            if d < 1e-12:
                # If distance is 0, push them apart randomly or skip
                continue
            v = pair_violations[idx]
            diff = x[i] - x[j]
            factor = -2 * v / d
            grad[i] += factor * diff
            grad[j] -= factor * diff
            
    # Wall gradient
    # d/dx (max(0, r - (0.5 - |x|))^2)
    # Let w = 0.5 - |x|
    # V = max(0, r - w)
    # If V > 0, dV/dx = -dw/dx
    # dw/dx = -sign(x)
    # So dV/dx = sign(x)
    # d(V^2)/dx = 2 * V * sign(x)
    
    active_wx = wall_viol_x > 1e-12
    if np.any(active_wx):
        for i in np.where(active_wx)[0]:
            v = wall_viol_x[i]
            sign_x = np.sign(x[i, 0])
            if sign_x == 0:
                sign_x = 1.0
            grad[i, 0] += 2 * v * sign_x
            
    active_wy = wall_viol_y > 1e-12
    if np.any(active_wy):
        for i in np.where(active_wy)[0]:
            v = wall_viol_y[i]
            sign_y = np.sign(x[i, 1])
            if sign_y == 0:
                sign_y = 1.0
            grad[i, 1] += 2 * v * sign_y
            
    return obj_val, grad

def _init_hex(r_guess=0.07):
    """Initialize points in a hexagonal pattern."""
    dx = 2 * r_guess
    dy = np.sqrt(3) * r_guess
    
    points = []
    y = -0.5 + dy/2
    row_idx = 0
    while y <= 0.5:
        x_start = -0.5 + dx/2
        if row_idx % 2 == 1:
            x_start += dx/2
        x = x_start
        while x <= 0.5:
            points.append((x, y))
            x += dx
        y += dy
        row_idx += 1
        
    points = np.array(points)
    
    if len(points) >= N:
        # Pick N points that are well-distributed
        # Simple random selection might be okay, but let's try to pick from the grid
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (remaining, 2))
        return np.vstack([points, random_pts])

def _try_feasibility(r, x_init, deadline):
    """
    Try to find a feasible configuration for radius r.
    Returns (success, x_opt)
    """
    if time.time() > deadline:
        return False, None
        
    # Use L-BFGS-B to minimize violation
    # We use a small number of restarts if needed, but here we just do one deep run
    # The init is expected to be good
    
    try:
        res = minimize(
            _violation_and_grad,
            x_init,
            args=(r,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        x_opt = res.x.reshape(N, 2)
        
        # Check if it's actually feasible (violation is very small)
        val, _ = _violation_and_grad(x_opt, r)
        
        # Also check if it's inside the square
        if np.any(x_opt < -0.5) or np.any(x_opt > 0.5):
            return False, None
            
        if val < 1e-8:
            return True, x_opt
        else:
            return False, None
            
    except Exception:
        return False, None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + (150.0 - _DEADLINE_MARGIN)
    
    best_x = None
    best_score = -np.inf
    
    # Initial guess for radius
    # The known best is ~0.071377
    # Let's start binary search between 0.070 and 0.075
    
    r_low = 0.070
    r_high = 0.075
    
    # First, ensure we have a valid starting configuration for r_low
    # Use hex grid
    x_hex = _init_hex(r_low)
    
    # Check if r_low is feasible with this init
    success, x_opt = _try_feasibility(r_low, x_hex, deadline)
    if success:
        best_x = x_opt
        best_score = _compute_radius(best_x)
    else:
        # If even 0.070 fails, try a simpler grid or random
        # But 0.070 should be easily feasible
        # Let's try a few random restarts for r_low
        for _ in range(5):
            if time.time() > deadline:
                break
            x_rand = np.random.uniform(-0.45, 0.45, (N, 2))
            success, x_opt = _try_feasibility(r_low, x_rand, deadline)
            if success:
                best_x = x_opt
                best_score = _compute_radius(best_x)
                break
                
    if best_x is None:
        # Fallback: return hex grid, even if not perfectly feasible, it's better than nothing
        best_x = x_hex
        best_score = _compute_radius(best_x)

    # Binary search for the maximum feasible radius
    # We perform a few iterations of binary search
    # Each iteration involves a feasibility check
    # To improve success rate, we can use the result of the previous successful step as the init for the next
    
    current_x = best_x.copy()
    
    # Number of binary search steps
    # 20 steps gives precision of (0.005) / 2^20 ~ 5e-9, which is plenty
    # But each step takes time. Let's do 15-20 steps.
    
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        # Use current_x as init, and also try a fresh hex init for r_mid
        # If current_x fails, try hex
        
        success = False
        x_candidate = None
        
        # Attempt 1: Use previous solution
        if time.time() < deadline - 2.0:
            success, x_candidate = _try_feasibility(r_mid, current_x, deadline)
            
        # Attempt 2: Use fresh hex init if previous failed or if we want diversity
        if not success and time.time() < deadline - 2.0:
            x_hex_mid = _init_hex(r_mid)
            success, x_candidate = _try_feasibility(r_mid, x_hex_mid, deadline)
            
        # Attempt 3: Random restart if still failing
        if not success and time.time() < deadline - 2.0:
            x_rand = np.random.uniform(-0.45, 0.45, (N, 2))
            success, x_candidate = _try_feasibility(r_mid, x_rand, deadline)
            
        if success:
            r_low = r_mid
            current_x = x_candidate
            # Update best score
            score = _compute_radius(x_candidate)
            if score > best_score:
                best_score = score
                best_x = x_candidate.copy()
        else:
            r_high = r_mid
            
    # Final Polish
    # Now that we have a good r_low and a configuration best_x,
    # we can try to maximize the radius directly using the soft-min approach
    # starting from best_x, to squeeze out any remaining gains.
    
    if time.time() < deadline - 1.0:
        # Use a soft-min objective for polishing
        # This handles the "min" structure directly
        
        def _softmin_obj_and_grad(x, beta):
            x = x.reshape(N, 2)
            dx = x[:, 0:1] - x[:, 0:1].T
            dy = x[:, 1:2] - x[:, 1:2].T
            dists_sq = dx**2 + dy**2
            np.fill_diagonal(dists_sq, 1.0)
            dists = np.sqrt(dists_sq)
            
            wx = 0.5 - np.abs(x[:, 0])
            wy = 0.5 - np.abs(x[:, 1])
            walls = np.minimum(wx, wy)
            
            iu = np.triu_indices(N, k=1)
            pair_dists = dists[iu]
            wall_vals = 2 * walls
            
            vals = np.concatenate([pair_dists, wall_vals])
            
            max_val = np.max(vals)
            exp_vals = np.exp(-beta * (vals - max_val))
            sum_exp = np.sum(exp_vals)
            softmin_val = max_val - (1.0 / beta) * np.log(sum_exp)
            
            obj_val = -softmin_val
            
            p = exp_vals / sum_exp
            n_pairs = len(pair_dists)
            pair_p = p[:n_pairs]
            wall_p = p[n_pairs:]
            
            g_soft = np.zeros_like(x)
            
            for idx in range(n_pairs):
                p_k = pair_p[idx]
                if p_k < 1e-15:
                    continue
                i, j = iu[0][idx], iu[1][idx]
                d = pair_dists[idx]
                if d < 1e-8:
                    continue
                diff = x[i] - x[j]
                g_soft[i] += p_k * (diff / d)
                g_soft[j] -= p_k * (diff / d)
                
            for i in range(N):
                p_k = wall_p[i]
                if p_k < 1e-15:
                    continue
                wx_i = wx[i]
                wy_i = wy[i]
                if wx_i < wy_i:
                    g_soft[i, 0] += p_k * 2.0 * (-np.sign(x[i, 0]))
                elif wy_i < wx_i:
                    g_soft[i, 1] += p_k * 2.0 * (-np.sign(x[i, 1]))
                else:
                    g_soft[i, 0] += p_k * 2.0 * (-np.sign(x[i, 0])) * 0.5
                    g_soft[i, 1] += p_k * 2.0 * (-np.sign(x[i, 1])) * 0.5
                    
            grad_obj = -g_soft
            return obj_val, grad_obj

        try:
            res = minimize(
                _softmin_obj_and_grad,
                best_x,
                args=(200.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_polished = res.x.reshape(N, 2)
            score_polished = _compute_radius(x_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished.copy()
        except Exception:
            pass

    return best_x
