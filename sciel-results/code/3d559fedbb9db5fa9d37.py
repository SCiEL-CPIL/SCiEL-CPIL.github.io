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

def _objective(params, s):
    """Calculate overlap and boundary violations for equal squares of side s."""
    n = 10
    centers = params.reshape((n, 2))
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Boundary violations
    bound_viol = 0.0
    for i in range(n):
        if x[i] - s/2 < 0:
            bound_viol += (x[i] - s/2)**2
        if x[i] + s/2 > 1:
            bound_viol += (x[i] + s/2 - 1)**2
        if y[i] - s/2 < 0:
            bound_viol += (y[i] - s/2)**2
        if y[i] + s/2 > 1:
            bound_viol += (y[i] + s/2 - 1)**2
            
    # Overlap violations
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(x[i] - x[j])
            dy = abs(y[i] - y[j])
            # If overlapping in both axes
            if dx < s and dy < s:
                # Use squared distance of "penetration"
                # Simple penalty: (s - dx)^2 + (s - dy)^2
                overlap_viol += (s - dx)**2 + (s - dy)**2
                
    return bound_viol + overlap_viol

def _pack_equal_squares(s, max_iter=2000):
    """Try to pack 10 equal squares of side s. Returns centers if valid, else None."""
    n = 10
    # Initial guess: 3x3 grid + 1 in corner, scaled to fit
    # Or just random
    best_centers = None
    best_score = np.inf
    
    # Try a few random initializations
    for attempt in range(5):
        if attempt == 0:
            # Grid initialization
            centers = np.zeros((n, 2))
            # 3x3 grid
            for i in range(9):
                r = i // 3
                c = i % 3
                centers[i, 0] = (c + 0.5) * s
                centers[i, 1] = (r + 0.5) * s
            # 10th in top right corner
            centers[9, 0] = 1 - s/2
            centers[9, 1] = 1 - s/2
        else:
            # Random initialization within [s/2, 1-s/2]
            centers = np.random.uniform(s/2, 1-s/2, (n, 2))
            
        params = centers.flatten()
        
        # Optimize
        res = minimize(_objective, params, args=(s,), method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_score:
            best_score = res.fun
            best_centers = res.x.reshape((n, 2))
            
        if best_score < 1e-8:
            break
            
    if best_score < 1e-8:
        # Final validation
        sides = np.full(n, s)
        if _valid(best_centers, sides):
            return best_centers, sides
    return None

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = 630.0 - 10.0
    
    # 1. Try to find a packing of 10 equal squares
    # Binary search for max s
    lo = 0.2
    hi = 0.32 # Slightly above 1/sqrt(10) ~ 0.3162
    
    best_equal_result = None
    best_equal_s = 0.0
    
    while hi - lo > 1e-4 and time.time() < deadline - 10:
        mid = (lo + hi) / 2
        res = _pack_equal_squares(mid)
        if res is not None:
            best_equal_result = res
            best_equal_s = mid
            lo = mid
        else:
            hi = mid
            
    if best_equal_result is not None:
        centers, sides = best_equal_result
        sum_sides = np.sum(sides)
        # Check if this is better than the constructive 3.0
        if sum_sides > 3.0:
            return centers, sides

    # 2. Fallback to constructive layout (3x3 grid + 1 small)
    s = 0.33333
    S = 0.00001
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
