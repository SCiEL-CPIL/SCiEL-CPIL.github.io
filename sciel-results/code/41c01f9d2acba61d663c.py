# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-END

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square to maximize the sum of side lengths.
    Uses a deterministic initial layout followed by gradient-based local optimization
    with a smooth penalty for overlaps and boundary violations.
    """
    start_time = time.time()
    DEADLINE = 620.0  # safety margin before 630s kill
    n = 10

    # ---- Initial layout: 4x4 grid, 10 squares ----
    # Use a 4x4 grid with side 0.25, but we'll optimize.
    # Initial positions: first 10 cells of a 4x4 grid
    k = 4
    side0 = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side0
    cy0 = (rows + 0.5) * side0
    s0 = np.full(n, side0, dtype=np.float64)
    
    # Convert to optimization variables: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    x0 = np.concatenate([np.column_stack([cx0, cy0, s0]).flatten()])
    
    # ---- Objective: minimize -sum(sides) + penalty ----
    def objective(x):
        if time.time() > DEADLINE:
            return 1e10  # signal timeout
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Boundary penalties (soft)
        # Each square must satisfy: cx - s/2 >= -tol, cx + s/2 <= 1+tol
        # i.e., s <= 2*(cx + tol) and s <= 2*(1 - cx + tol)
        # and similarly for y
        tol = 1e-7
        margin_x = np.minimum(cx + tol, 1.0 - cx + tol)
        margin_y = np.minimum(cy + tol, 1.0 - cy + tol)
        margin = np.minimum(margin_x, margin_y)
        # If s > 2*margin, penalty
        over = np.maximum(s - 2.0 * margin, 0.0)
        boundary_pen = 100.0 * np.sum(over**2)
        
        # Overlap penalties
        # Squares i and j overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
        # We use a smooth penalty: for each pair, compute the "separation" in x and y
        # sep_x = |cx_i - cx_j| - (s_i + s_j)/2 + tol
        # sep_y = |cy_i - cy_j| - (s_i + s_j)/2 + tol
        # If both sep_x < 0 and sep_y < 0, they overlap.
        # Penalty: max(0, -min(sep_x, sep_y)) ** 2  -- this is 0 if separated in either axis
        # Actually, if they overlap, both sep_x and sep_y are negative. The depth of overlap
        # is min(-sep_x, -sep_y). We want to penalize when min(sep_x, sep_y) < 0.
        # Let d = min(sep_x, sep_y). If d < 0, penalty = d^2.
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx[i] - cx[j]) - (s[i] + s[j]) / 2.0 + tol
                dy = abs(cy[i] - cy[j]) - (s[i] + s[j]) / 2.0 + tol
                d = min(dx, dy)
                if d < 0:
                    overlap_pen += d * d
        overlap_pen *= 100.0
        
        # Also penalize negative sides
        neg_s = np.maximum(-s, 0.0)
        neg_pen = 1000.0 * np.sum(neg_s**2)
        
        total_pen = boundary_pen + overlap_pen + neg_pen
        return -np.sum(s) + total_pen
    
    def gradient(x):
        # Numerical gradient (central difference)
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            fp = objective(xp)
            fm = objective(xm)
            grad[i] = (fp - fm) / (2.0 * eps)
        return grad

    # ---- Optimization ----
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try SLSQP with constraints
    # Constraints: s_i > 0 (handled by bounds), boundary, no overlap
    # For SLSQP, we use inequality constraints:
    # For each square i: s_i - 2*(cx_i + tol) <= 0, s_i - 2*(1-cx_i+tol) <= 0, etc.
    # This is complex. Let's use a simpler approach: multiple restarts with Nelder-Mead or SLSQP with penalty.
    
    # Use SLSQP with a few constraints for boundary
    constraints = []
    for i in range(n):
        # s_i <= 2*(cx_i + tol)
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2.0 * (x[3*i] + tol) - x[3*i+2]})
        # s_i <= 2*(1 - cx_i + tol)
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2.0 * (1.0 - x[3*i] + tol) - x[3*i+2]})
        # s_i <= 2*(cy_i + tol)
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2.0 * (x[3*i+1] + tol) - x[3*i+2]})
        # s_i <= 2*(1 - cy_i + tol)
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2.0 * (1.0 - x[3*i+1] + tol) - x[3*i+2]})
    
    bounds = [(0.01, 1.0)] * n + [(0.01, 1.0)] * n + [(0.01, 1.0)] * n
    # cx, cy in [0.01, 0.99], s in [0.01, 1.0]
    bounds = [(0.001, 0.999)] * n + [(0.001, 0.999)] * n + [(0.01, 1.0)] * n

    # Run SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-9})
        if time.time() < DEADLINE:
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass

    # Refine with Nelder-Mead
    if time.time() < DEADLINE:
        try:
            res2 = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Extract result
    cx = best_x[0::3]
    cy = best_x[1::3]
    s = best_x[2::3]
    
    # Clamp to valid range
    tol = 1e-7
    cx = np.clip(cx, tol, 1.0 - tol)
    cy = np.clip(cy, tol, 1.0 - tol)
    s = np.clip(s, 1e-6, 1.0)
    
    # Ensure s doesn't exceed boundary
    for i in range(n):
        max_s = 2.0 * min(cx[i] + tol, 1.0 - cx[i] + tol, cy[i] + tol, 1.0 - cy[i] + tol)
        s[i] = min(s[i], max_s)
    
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-START
import numpy as np


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].

    Baseline construction: a simple deterministic grid packing. With
    k = ceil(sqrt(n)) columns/rows and side = 1/k, the first n grid cells each
    hold one square. This is feasible (squares only touch, never overlap, and
    stay inside the unit square) but suboptimal.
    """
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


# EVOLVE-BLOCK-END
