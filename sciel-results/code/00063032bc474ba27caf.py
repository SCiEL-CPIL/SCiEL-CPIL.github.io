# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _valid(centers, sides, tol=1e-7):
    """Check validity of a packing configuration."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if not np.all(sides > 0):
        return False
    # Boundary check
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Non-overlap check
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _softmax_penalty(v, beta=50.0):
    """Smooth approximation of max(0, v) using log-sum-exp."""
    # For v > 0: approximately v; for v < 0: approximately 0
    # Use: (1/beta) * log(1 + exp(beta*v)) - v/beta for v<0 region
    # Simpler: use softplus-like: (1/beta)*log(1+exp(beta*v)) but shift
    # Actually, we want penalty for violation, so penalty = max(0, v)^2 smoothed
    # Use: (1/(2*beta)) * (exp(beta*v) - 1) for v >= 0, 0 for v < 0 (approx)
    # Better: use smooth max(0,v) = (v + sqrt(v^2 + eps))/2
    eps = 1e-10
    return (v + np.sqrt(v**2 + eps)) / 2.0

def _objective_smooth(params, s, beta=20.0):
    """Calculate smooth overlap and boundary violations for equal squares of side s."""
    n = 10
    centers = params.reshape((n, 2))
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Boundary violations: penalty when square extends outside [0,1]
    bound_viol = 0.0
    for i in range(n):
        # Left boundary
        v_left = -(x[i] - s/2)  # positive if violation
        bound_viol += _softmax_penalty(v_left, beta) ** 2
        # Right boundary
        v_right = x[i] + s/2 - 1
        bound_viol += _softmax_penalty(v_right, beta) ** 2
        # Bottom boundary
        v_bottom = -(y[i] - s/2)
        bound_viol += _softmax_penalty(v_bottom, beta) ** 2
        # Top boundary
        v_top = y[i] + s/2 - 1
        bound_viol += _softmax_penalty(v_top, beta) ** 2
        
    # Overlap violations: for each pair, if both dx < s and dy < s, penalize
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(x[i] - x[j])
            dy = abs(y[i] - y[j])
            # Violation if dx < s AND dy < s
            # Smooth: penalty = max(0, s-dx) * max(0, s-dy)
            pen_x = _softmax_penalty(s - dx, beta)
            pen_y = _softmax_penalty(s - dy, beta)
            overlap_viol += pen_x * pen_y
            
    return bound_viol + overlap_viol

def _pack_equal_squares(s, max_iter=1500, n_restarts=3):
    """Try to pack 10 equal squares of side s. Returns centers if valid, else None."""
    n = 10
    best_centers = None
    best_score = np.inf
    
    # Known good initial patterns
    def get_initial_centers():
        patterns = []
        
        # Pattern 0: 3x3 grid + 1 in corner
        c0 = np.zeros((n, 2))
        for i in range(9):
            r = i // 3
            c = i % 3
            c0[i, 0] = (c + 0.5) * s
            c0[i, 1] = (r + 0.5) * s
        c0[9, 0] = 1 - s/2
        c0[9, 1] = 1 - s/2
        patterns.append(c0)
        
        # Pattern 1: Random
        c1 = np.random.uniform(s/2, 1-s/2, (n, 2))
        patterns.append(c1)
        
        # Pattern 2: Two rows of 5, slightly offset
        c2 = np.zeros((n, 2))
        for i in range(5):
            c2[i, 0] = (i + 0.5) * (1.0/5) * s * 1.1  # spread out
            c2[i, 1] = 0.25
            c2[i+5, 0] = (i + 0.5) * (1.0/5) * s * 1.1
            c2[i+5, 1] = 0.75
        # Clip to valid range
        c2 = np.clip(c2, s/2, 1-s/2)
        patterns.append(c2)
        
        # Pattern 3: Random with larger spread
        c3 = np.random.uniform(s/2, 1-s/2, (n, 2))
        patterns.append(c3)
        
        return patterns
    
    patterns = get_initial_centers()
    
    for attempt in range(min(n_restarts, len(patterns))):
        params = patterns[attempt].flatten()
        
        # Optimize with smooth objective
        res = minimize(_objective_smooth, params, args=(s,), method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-12})
        
        if res.fun < best_score:
            best_score = res.fun
            best_centers = res.x.reshape((n, 2))
            
        if best_score < 1e-10:
            break
            
    if best_score < 1e-8:
        # Final validation with tighter tolerance
        sides = np.full(n, s)
        if _valid(best_centers, sides, tol=1e-7):
            return best_centers, sides
    return None

def _polish_packing(centers, s, max_iter=500):
    """Final high-precision polish pass for a feasible packing."""
    n = 10
    params = centers.flatten()
    
    # Use SLSQP with constraints for final polish
    # Constraints: each center must be in [s/2, 1-s/2]^2
    bounds = [(s/2, 1-s/2)] * (2*n)
    
    # Non-overlap constraints (linearized)
    # For each pair, at least one of: x_i - x_j >= s, x_j - x_i >= s, y_i - y_j >= s, y_j - y_i >= s
    # This is non-convex, so we'll just do a local optimization
    
    res = minimize(_objective_smooth, params, args=(s,), method='Nelder-Mead', 
                   bounds=bounds,
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-14})
    
    polished_centers = res.x.reshape((n, 2))
    sides = np.full(n, s)
    
    if _valid(polished_centers, sides, tol=1e-7):
        return polished_centers
    return centers

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = 630.0 - 15.0  # Safety margin
    
    # 1. Try to find a packing of 10 equal squares
    # Binary search for max s
    lo = 0.25
    hi = 0.317  # Slightly above 1/sqrt(10) ~ 0.3162
    
    best_equal_result = None
    best_equal_s = 0.0
    
    # Coarse binary search first
    for _ in range(8):
        if time.time() > deadline - 30:
            break
        mid = (lo + hi) / 2
        res = _pack_equal_squares(mid, max_iter=800, n_restarts=2)
        if res is not None:
            best_equal_result = res
            best_equal_s = mid
            lo = mid
        else:
            hi = mid
    
    # Fine binary search if we have time
    if time.time() < deadline - 20 and best_equal_result is not None:
        lo2 = best_equal_s
        hi2 = hi
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            mid = (lo2 + hi2) / 2
            res = _pack_equal_squares(mid, max_iter=1200, n_restarts=3)
            if res is not None:
                best_equal_result = res
                best_equal_s = mid
                lo2 = mid
            else:
                hi2 = mid
    
    if best_equal_result is not None:
        centers, sides = best_equal_result
        # Polish the solution
        if time.time() < deadline - 10:
            polished = _polish_packing(centers, sides[0], max_iter=300)
            if polished is not None:
                centers = polished
        
        sum_sides = np.sum(sides)
        # Check if this is better than the constructive 3.0
        if sum_sides > 3.0:
            return centers, sides

    # 2. Fallback to constructive layout (3x3 grid + 1 small)
    s = 1.0 / 3.0
    S = 1e-5
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    idx = 0
    for row in range(3):
        for col in range(3):
            centers[idx, 0] = (col + 0.5) * s
            centers[idx, 1] = (row + 0.5) * s
            sides[idx] = s
            idx += 1
            
    centers[idx, 0] = 1.0 - S / 2
    centers[idx, 1] = 1.0 - S / 2
    sides[idx] = S
    
    if not _valid(centers, sides):
        # Final fallback: 4x4 grid with 10 squares
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        
    return centers, sides

# EVOLVE-BLOCK-END
