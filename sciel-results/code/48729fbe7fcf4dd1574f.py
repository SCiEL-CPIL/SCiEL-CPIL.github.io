import numpy as np
from scipy.optimize import minimize
import time

def _smooth_max(a, b, beta=50.0):
    """Smooth maximum of a and b."""
    return np.logaddexp(beta * a, beta * b) / beta

def _smooth_overlap_penalty(x, y, s, beta=50.0):
    """
    Compute smooth overlap penalty for all pairs.
    For squares i,j: overlap exists if |cx_i-cx_j| < (s_i+s_j)/2 AND |cy_i-cy_j| < (s_i+s_j)/2
    We use smooth max to create a differentiable penalty.
    """
    n = len(s)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            req = (s[i] + s[j]) / 2.0
            dx = abs(x[i] - x[j])
            dy = abs(y[i] - y[j])
            # Overlap in x: req - dx (positive if overlapping)
            ox = _smooth_max(req - dx, 0.0, beta)
            oy = _smooth_max(req - dy, 0.0, beta)
            penalty += ox * oy
    return penalty

def _boundary_penalty(x, y, s, beta=50.0):
    """Smooth boundary violation penalty."""
    n = len(s)
    p = 0.0
    for i in range(n):
        half = s[i] / 2.0
        # Left: x - half < 0 => half - x > 0
        p += _smooth_max(half - x[i], 0.0, beta) ** 2
        # Right: x + half > 1 => x + half - 1 > 0
        p += _smooth_max(x[i] + half - 1.0, 0.0, beta) ** 2
        # Bottom
        p += _smooth_max(half - y[i], 0.0, beta) ** 2
        # Top
        p += _smooth_max(y[i] + half - 1.0, 0.0, beta) ** 2
    return p

def _objective(v, beta=50.0):
    n = 10
    v = v.reshape(n, 3)
    x = v[:, 0]
    y = v[:, 1]
    s = v[:, 2]
    
    sum_s = np.sum(s)
    P_o = _smooth_overlap_penalty(x, y, s, beta)
    P_b = _boundary_penalty(x, y, s, beta)
    
    lam = 200.0
    return -sum_s + lam * (P_o + P_b)

def _is_valid(v):
    n = 10
    v = v.reshape(n, 3)
    x = v[:, 0]
    y = v[:, 1]
    s = v[:, 2]
    
    tol = 1e-7
    if np.any(s <= 0):
        return False
    if np.any(~np.isfinite(v)):
        return False
        
    # Boundary check
    if np.any(x - s/2 < -tol) or np.any(x + s/2 > 1 + tol):
        return False
    if np.any(y - s/2 < -tol) or np.any(y + s/2 > 1 + tol):
        return False
        
    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(x[i] - x[j])
            dy = abs(y[i] - y[j])
            t = (s[i] + s[j]) / 2 - tol
            if dx < t and dy < t:
                return False
    return True

def _make_known_packing():
    """
    Known good packing: 2 large squares (side ~0.5) and 8 small squares.
    Layout:
    - Large square 1: center (0.25, 0.25), side 0.5
    - Large square 2: center (0.75, 0.75), side 0.5
    - 8 small squares filling remaining space
    """
    n = 10
    x = np.zeros(n)
    y = np.zeros(n)
    s = np.zeros(n)
    
    # Two large squares
    x[0], y[0], s[0] = 0.25, 0.25, 0.5
    x[1], y[1], s[1] = 0.75, 0.75, 0.5
    
    # 8 small squares in the remaining L-shaped region
    # Top-left region: [0, 0.5] x [0.5, 1] -> 2x2 grid of 0.25
    # Bottom-right region: [0.5, 1] x [0, 0.5] -> 2x2 grid of 0.25
    # Middle cross regions need careful placement
    
    # Let's use a different known configuration:
    # 2 large (0.5) at corners, 8 small (0.25) in a 4x2 arrangement
    # Actually, let me use the standard 2+8 packing:
    # Large at (0.25, 0.25) and (0.75, 0.75)
    # Small squares of side 0.25:
    # (0.25, 0.75), (0.75, 0.25) - diagonal from large
    # (0.125, 0.125), (0.375, 0.125), (0.125, 0.375), (0.375, 0.375) - but these overlap with large!
    
    # Better: use the configuration where 2 large are in opposite corners
    # and 8 small fill the rest in a 4x2 strip pattern
    # Large 1: [0, 0.5] x [0, 0.5], center (0.25, 0.25)
    # Large 2: [0.5, 1] x [0.5, 1], center (0.75, 0.75)
    # Remaining: [0, 0.5] x [0.5, 1] and [0.5, 1] x [0, 0.5]
    # Each is 0.5 x 0.5, can fit 2x2 = 4 small squares of side 0.25
    # Total: 4 + 4 = 8 small squares
    
    # Top-left quadrant small squares (in region [0,0.5]x[0.5,1])
    positions_tl = [
        (0.125, 0.625),
        (0.375, 0.625),
        (0.125, 0.875),
        (0.375, 0.875),
    ]
    for k, (px, py) in enumerate(positions_tl):
        x[2 + k], y[2 + k], s[2 + k] = px, py, 0.25
    
    # Bottom-right quadrant small squares (in region [0.5,1]x[0,0.5])
    positions_br = [
        (0.625, 0.125),
        (0.875, 0.125),
        (0.625, 0.375),
        (0.875, 0.375),
    ]
    for k, (px, py) in enumerate(positions_br):
        x[6 + k], y[6 + k], s[6 + k] = px, py, 0.25
    
    return np.column_stack([x, y, s]).flatten()

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 585  # 45s safety margin
    
    # Initial guesses - prioritize known good packings
    inits = []
    
    # 1. Known 2-large + 8-small packing (sum = 2*0.5 + 8*0.25 = 3.0)
    inits.append(_make_known_packing())
    
    # 2. Slight perturbations of known packing for local refinement
    rng = np.random.default_rng(42)
    base = _make_known_packing().copy()
    for _ in range(3):
        v_pert = base + rng.normal(0, 0.02, size=base.shape)
        inits.append(v_pert)
    
    # 3. Alternative: 1 large + 9 small
    # Large at center (0.5, 0.5) side 0.5, 9 small around it
    x = np.zeros(n)
    y = np.zeros(n)
    s = np.zeros(n)
    x[0], y[0], s[0] = 0.5, 0.5, 0.5
    # 9 small of side ~0.2 in surrounding positions
    small_pos = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5),         (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
        (0.5, 0.5),  # center - will be adjusted
    ]
    for k in range(1, n):
        x[k], y[k] = small_pos[k-1]
        s[k] = 0.2
    inits.append(np.column_stack([x, y, s]).flatten())
    
    # 4. Uniform grid 4x4 (10 squares, side 0.25)
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    s_init = np.full(n, side)
    inits.append(np.column_stack([cx, cy, s_init]).flatten())
    
    # 5. Random initializations
    for _ in range(5):
        x = rng.uniform(0.1, 0.9, n)
        y = rng.uniform(0.1, 0.9, n)
        s = rng.uniform(0.15, 0.4, n)
        inits.append(np.column_stack([x, y, s]).flatten())
    
    bounds = [(0, 1), (0, 1), (1e-6, 1)] * n
    
    best_sum = 0.0
    best_v = None
    
    # Multi-start with smooth objective
    for v0 in inits:
        if time.time() > deadline - 10:
            break
        
        try:
            # Use L-BFGS-B with smooth objective
            res = minimize(_objective, v0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            
            if _is_valid(res.x):
                v = res.x.reshape(n, 3)
                curr_sum = np.sum(v[:, 2])
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead for local refinement if we have a good candidate
        if best_v is not None and time.time() < deadline - 15:
            try:
                res_nm = minimize(_objective, best_v, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
                if _is_valid(res_nm.x):
                    v = res_nm.x.reshape(n, 3)
                    curr_sum = np.sum(v[:, 2])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_v = res_nm.x.copy()
            except Exception:
                pass
    
    # Fallback: return the known packing directly
    if best_v is None:
        v = _make_known_packing().reshape(n, 3)
        centers = v[:, :2].astype(np.float64)
        sides = v[:, 2].astype(np.float64)
        return centers, sides
    
    v = best_v.reshape(n, 3)
    centers = v[:, :2].astype(np.float64)
    sides = v[:, 2].astype(np.float64)
    
    return centers, sides
