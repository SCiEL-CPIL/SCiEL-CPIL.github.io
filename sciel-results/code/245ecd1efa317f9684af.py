# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    # Check shapes
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    # Check finite and positive
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(x, n=10):
    """Objective: negative sum of side lengths (we minimize)."""
    # x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    centers = x.reshape(n, 3)[:, :2]
    sides = x.reshape(n, 3)[:, 2]
    return -np.sum(sides)


def _constraints(x, n=10, tol=1e-7):
    """Generate constraint functions for SLSQP."""
    cons = []
    centers = x.reshape(n, 3)[:, :2]
    sides = x.reshape(n, 3)[:, 2]
    
    # Boundary constraints: each square must be inside [0,1]^2
    for i in range(n):
        # cx - s/2 >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]/2 - 0})
        # cx + s/2 <= 1
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]/2})
        # cy - s/2 >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]/2})
        # cy + s/2 <= 1
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]/2})
        # s > 0 (use small lower bound)
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - 0.001})
    
    # Non-overlap constraints: for each pair, at least one axis must be separated
    # |cx_i - cx_j| >= (s_i + s_j)/2 - tol  OR  |cy_i - cy_j| >= (s_i + s_j)/2 - tol
    # This is a disjunctive constraint. We handle it by adding both as "soft" 
    # constraints using a penalty approach or by using the fact that SLSQP can
    # handle it if we add both as ineq (which would be wrong).
    # Better: use a penalty method or add constraints that ensure separation.
    # For SLSQP, we can add the constraint that the minimum of the two separations
    # is >= 0, but that's not differentiable.
    # Alternative: add both constraints as ineq, which forces BOTH axes to be separated
    # (stronger than needed, but feasible).
    for i in range(n):
        for j in range(i+1, n):
            t = 0  # will be computed inside
            # |cx_i - cx_j| + tol >= (s_i + s_j)/2
            # This is equivalent to: cx_i - cx_j >= (s_i+s_j)/2 - tol OR cx_j - cx_i >= (s_i+s_j)/2 - tol
            # Add both as separate ineq constraints (this is overly restrictive but valid)
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                         x[3*i] - x[3*j] - (x[3*i+2] + x[3*j+2])/2 + tol})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                         x[3*j] - x[3*i] - (x[3*i+2] + x[3*j+2])/2 + tol})
            # Same for y
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                         x[3*i+1] - x[3*j+1] - (x[3*i+2] + x[3*j+2])/2 + tol})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                         x[3*j+1] - x[3*i+1] - (x[3*i+2] + x[3*j+2])/2 + tol})
    
    return cons


def _grid_baseline(n=10):
    """Generate the grid baseline."""
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
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600  # seconds
    
    # Get baseline
    centers_base, sides_base = _grid_baseline(n)
    best_centers = centers_base.copy()
    best_sides = sides_base.copy()
    best_value = np.sum(sides_base)
    
    # Try optimization with multiple restarts
    rng = np.random.default_rng(42)
    
    # Initial guess from grid
    x0_base = np.zeros(n * 3)
    for i in range(n):
        x0_base[3*i] = centers_base[i, 0]
        x0_base[3*i+1] = centers_base[i, 1]
        x0_base[3*i+2] = sides_base[i]
    
    bounds = [(0, 1), (0, 1), (0.001, 1.0)] * n
    
    # Try SLSQP from grid baseline
    try:
        cons = _constraints(x0_base, n, tol)
        result = minimize(_objective, x0_base, method='SLSQP', bounds=bounds,
                         constraints=cons, options={'maxiter': 500, 'ftol': 1e-10})
        if result.success or result.fun < -best_value:
            x_opt = result.x
            c = x_opt.reshape(n, 3)[:, :2]
            s = x_opt.reshape(n, 3)[:, 2]
            if _is_valid(c, s, tol):
                val = np.sum(s)
                if val > best_value:
                    best_value = val
                    best_centers = c.copy()
                    best_sides = s.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 60:
        return best_centers, best_sides
    
    # Try random restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - 30:
            break
        
        # Random initialization near grid
        x0 = x0_base.copy()
        noise_scale = 0.1
        for i in range(n):
            x0[3*i] += rng.uniform(-noise_scale, noise_scale)
            x0[3*i+1] += rng.uniform(-noise_scale, noise_scale)
            x0[3*i+2] = max(0.001, x0[3*i+2] + rng.uniform(-0.1, 0.1))
        
        # Clip to bounds
        for i in range(n):
            x0[3*i] = np.clip(x0[3*i], 0, 1)
            x0[3*i+1] = np.clip(x0[3*i+1], 0, 1)
            x0[3*i+2] = np.clip(x0[3*i+2], 0.001, 1.0)
        
        try:
            cons = _constraints(x0, n, tol)
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 300, 'ftol': 1e-10})
            if result.success or result.fun < -best_value:
                x_opt = result.x
                c = x_opt.reshape(n, 3)[:, :2]
                s = x_opt.reshape(n, 3)[:, 2]
                if _is_valid(c, s, tol):
                    val = np.sum(s)
                    if val > best_value:
                        best_value = val
                        best_centers = c.copy()
                        best_sides = s.copy()
        except Exception:
            pass
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
