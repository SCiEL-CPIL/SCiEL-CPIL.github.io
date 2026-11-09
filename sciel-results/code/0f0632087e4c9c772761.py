import numpy as np
import time
import scipy.optimize

def _is_valid_vectorized(centers, sides, tol=1e-7):
    """
    Vectorized validity check.
    centers: (N, 2)
    sides: (N,)
    """
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    
    # Required separation
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    
    # Check for any pair (i, j) where i != j that violates separation
    mask = ~np.eye(n, dtype=bool)
    
    # Overlap condition: dx < sep AND dy < sep
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _initial_packing():
    """
    Returns the known good packing with sum 3.0.
    2 squares of 0.5, 8 squares of 0.25.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # 2 Large at bottom
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    # 8 Small in top half (y in [0.5, 1.0])
    # Two rows of 4
    for k in range(8):
        row = k // 4  # 0 or 1
        col = k % 4   # 0 to 3
        
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        
        centers[2+k] = [cx, cy]
        sides[2+k] = 0.25
        
    return centers, sides

def _penalty_objective(x, n=10, tol=1e-7):
    """
    Objective function for scipy.optimize.minimize.
    Minimizes -sum(sides) + penalty for violations.
    x is a flat array of length 3*n: [cx0, cy0, s0, cx1, cy1, s1, ...]
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Objective: maximize sum(sides) -> minimize -sum(sides)
    obj = -np.sum(sides)
    
    # Penalty for boundary violations
    # We want centers[i] +/- sides[i]/2 in [0, 1]
    # Violation if center - s/2 < 0 or center + s/2 > 1
    # Smooth penalty: max(0, - (center - s/2))^2 + max(0, (center + s/2 - 1))^2
    # But we have tol. Let's use strict boundaries for penalty to keep it inside.
    # Actually, the validity check allows [-tol, 1+tol].
    # To be safe, let's penalize going outside [0, 1].
    
    # x-boundaries
    left_viol = np.maximum(0, -(centers[:, 0] - sides/2))
    right_viol = np.maximum(0, (centers[:, 0] + sides/2 - 1))
    # y-boundaries
    bottom_viol = np.maximum(0, -(centers[:, 1] - sides/2))
    top_viol = np.maximum(0, (centers[:, 1] + sides/2 - 1))
    
    bound_pen = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Penalty for overlap
    # Overlap if |dx| < (si+sj)/2 - tol AND |dy| < (si+sj)/2 - tol
    # Let's define separation margin
    # sep_req = (si+sj)/2 - tol
    # dx = |cx_i - cx_j|
    # dy = |cy_i - cy_j|
    # Violation is if dx < sep_req AND dy < sep_req.
    # We can approximate the "AND" by penalizing the minimum of the margins.
    # margin_x = dx - sep_req
    # margin_y = dy - sep_req
    # If both > 0, no violation.
    # If one < 0, no violation (separated in that axis).
    # If both < 0, violation.
    # A smooth approximation for the condition "both < 0" is to use a soft-min of the margins?
    # Or simply penalize the product of the negative parts?
    # Let's use: if margin_x < 0 and margin_y < 0, penalty = (min(margin_x, margin_y))^2?
    # Actually, if they overlap, we want to push them apart.
    # The "depth" of overlap in x is max(0, sep_req - dx).
    # The "depth" of overlap in y is max(0, sep_req - dy).
    # If both are > 0, they overlap.
    # Penalty = (min(depth_x, depth_y))^2? No, that's not right.
    # If depth_x=0.1, depth_y=0.1, overlap area is roughly 0.01.
    # If depth_x=0.1, depth_y=0.01, overlap area is smaller.
    # A simple penalty that works well is sum of (max(0, sep_req - dx) * max(0, sep_req - dy)).
    # This is 0 if separated in either axis.
    # It is positive if overlapping.
    
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep_req = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    
    # Depth of penetration
    pen_x = np.maximum(0, sep_req - dx)
    pen_y = np.maximum(0, sep_req - dy)
    
    # Penalty for each pair
    # Only consider i < j
    i_idx, j_idx = np.triu_indices(n, 1)
    pair_pen = pen_x[i_idx, j_idx] * pen_y[i_idx, j_idx]
    overlap_pen = np.sum(pair_pen**2) # Square to make it smoother/stronger
    
    # Also penalize negative sides
    neg_side_pen = np.sum(np.maximum(0, -sides)**2)
    
    # Weight the penalties
    # We want the penalty to be strong enough to prevent violations,
    # but not so strong that it dominates the objective.
    # Since sides are ~0.1-0.5, sum is ~3.
    # Penalties should be comparable.
    weight = 10.0
    
    return obj + weight * (bound_pen + overlap_pen + neg_side_pen)

def _polish_solution(centers, sides, max_time=60):
    """
    Uses scipy.optimize.minimize to polish the solution.
    Returns (centers, sides) or None if failed.
    """
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])
    
    # Define bounds
    # sides > 0, < 1
    # centers depend on sides, but for SLSQP we can set loose bounds
    # cx in [0, 1], cy in [0, 1], s in [0.01, 1.0]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s
    
    start_time = time.time()
    
    try:
        res = scipy.optimize.minimize(
            _penalty_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'maxfun': 1000}
        )
        
        if time.time() - start_time > max_time:
            return None
            
        if res.success or (not res.success and res.fun < _penalty_objective(x0)):
            x_opt = res.x
            new_centers = x_opt[:2*n].reshape(n, 2)
            new_sides = x_opt[2*n:]
            
            # Check validity
            if _is_valid_vectorized(new_centers, new_sides, tol=1e-7):
                return new_centers, new_sides
            else:
                # Try to repair by clipping
                # This is a heuristic repair
                for i in range(n):
                    s = new_sides[i]
                    new_centers[i, 0] = np.clip(new_centers[i, 0], s/2, 1 - s/2)
                    new_centers[i, 1] = np.clip(new_centers[i, 1], s/2, 1 - s/2)
                if _is_valid_vectorized(new_centers, new_sides, tol=1e-7):
                    return new_centers, new_sides
                else:
                    return None
        else:
            return None
            
    except Exception:
        return None

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 630
    
    # Initial configuration
    centers, sides = _initial_packing()
    
    # Verify initial
    if not _is_valid_vectorized(centers, sides, tol):
        centers = np.zeros((n, 2))
        sides = np.full(n, 0.25)
        for i in range(n):
            row = i // 4
            col = i % 4
            centers[i] = [0.125 + col*0.25, 0.125 + row*0.25]
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    current_val = best_val
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing Parameters
    T = 0.05
    cooling_rate = 0.9999
    min_T = 1e-8
    max_iters = 20_000_000
    
    # Run SA for first 300 seconds
    sa_deadline = start_time + 300
    
    for it in range(max_iters):
        if time.time() > sa_deadline:
            break
            
        if T < min_T:
            break
            
        new_centers = centers.copy()
        new_sides = sides.copy()
        
        idx = rng.integers(0, n)
        
        noise = 0.05
        new_centers[idx, 0] += rng.uniform(-noise, noise)
        new_centers[idx, 1] += rng.uniform(-noise, noise)
        
        size_noise = 0.05
        new_sides[idx] += rng.uniform(-size_noise, size_noise)
        
        s = new_sides[idx]
        if s <= 0.01:
            new_sides[idx] = 0.01
        if s > 1.0:
            new_sides[idx] = 1.0
        s = new_sides[idx]
        
        new_centers[idx, 0] = np.clip(new_centers[idx, 0], s/2, 1 - s/2)
        new_centers[idx, 1] = np.clip(new_centers[idx, 1], s/2, 1 - s/2)
        
        if _is_valid_vectorized(new_centers, new_sides, tol):
            new_val = np.sum(new_sides)
            delta = new_val - current_val
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                centers = new_centers
                sides = new_sides
                current_val = new_val
                
                if current_val > best_val:
                    best_val = current_val
                    best_centers = centers.copy()
                    best_sides = sides.copy()
                    
        T *= cooling_rate

    # Polish the best solution found
    # Use remaining time for polishing
    remaining_time = deadline - time.time() - 5 # 5s safety margin
    if remaining_time > 10:
        polished = _polish_solution(best_centers, best_sides, max_time=remaining_time)
        if polished is not None:
            p_centers, p_sides = polished
            p_val = np.sum(p_sides)
            if p_val > best_val:
                best_val = p_val
                best_centers = p_centers
                best_sides = p_sides
                
    return best_centers, best_sides
