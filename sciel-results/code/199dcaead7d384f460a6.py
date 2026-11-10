# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            if dx < t and dy < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _grow_square_max(centers, sides, i, tol=1e-7):
    """Find the maximum side length for square i given all other positions/sizes."""
    lo = sides[i]
    hi = 1.0
    cx, cy = centers[i]
    for _ in range(60):
        mid = (lo + hi) / 2.0
        half = mid / 2.0
        if cx - half < -tol or cx + half > 1.0 + tol:
            hi = mid
            continue
        if cy - half < -tol or cy + half > 1.0 + tol:
            hi = mid
            continue
        feasible = True
        for j in range(len(sides)):
            if j == i:
                continue
            t = (mid + sides[j]) / 2.0 - tol
            dx = abs(cx - centers[j, 0])
            dy = abs(cy - centers[j, 1])
            if dx < t and dy < t:
                feasible = False
                break
        if feasible:
            lo = mid
        else:
            hi = mid
    return lo


def _grow_all_iteratively(centers, sides, tol=1e-7, max_iters=300):
    """Iteratively grow each square to its maximum given others, repeating until convergence."""
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _grow_square_max(centers, sides, i, tol)
            if new_s > sides[i] + 1e-12:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _force_based_separation(centers, sides, tol=1e-7, max_iters=300, lr=0.05):
    """Use smooth repulsion forces to separate overlapping squares."""
    n = len(sides)
    for it in range(max_iters):
        forces = np.zeros((n, 2))
        max_overlap = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                ax = abs(dx)
                ay = abs(dy)
                
                if ax < t and ay < t:
                    overlap_x = max(0, t - ax)
                    overlap_y = max(0, t - ay)
                    max_overlap = max(max_overlap, overlap_x, overlap_y)
                    
                    if overlap_x < overlap_y:
                        sign = 1 if dx >= 0 else -1
                        forces[i, 0] += sign * overlap_x * 10
                        forces[j, 0] -= sign * overlap_x * 10
                    else:
                        sign = 1 if dy >= 0 else -1
                        forces[i, 1] += sign * overlap_y * 10
                        forces[j, 1] -= sign * overlap_y * 10
        
        if max_overlap < 1e-10:
            break
        
        for i in range(n):
            centers[i, 0] += lr * forces[i, 0]
            centers[i, 1] += lr * forces[i, 1]
        
        centers = _clamp_to_unit(centers, sides)
    
    return centers


def _voronoi_limited_side(centers, sides, i, tol=1e-7):
    """
    Compute the maximum side length square i can have given its position,
    limited by boundaries and all other squares.
    This is the 'Voronoi-limited' side: the min over all constraints.
    """
    cx, cy = centers[i]
    # Boundary limits
    max_from_boundary = min(cx, 1.0 - cx, cy, 1.0 - cy) * 2.0
    
    # Limits from each other square j
    # For square i and j to not overlap: |dx| >= (s_i + s_j)/2 OR |dy| >= (s_i + s_j)/2
    # This means: s_i <= 2*|dx| - s_j  (if separated in x)
    #              s_i <= 2*|dy| - s_j  (if separated in y)
    # The constraint is: NOT (|dx| < (s_i+s_j)/2 AND |dy| < (s_i+s_j)/2)
    # Equivalent: (s_i <= 2|dx| - s_j) OR (s_i <= 2|dy| - s_j)
    # So s_i <= max(2|dx| - s_j, 2|dy| - s_j)
    # But we also need s_i >= 0, and the bound is the min over j of these max values
    
    limit = max_from_boundary
    for j in range(len(sides)):
        if j == i:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        # s_i must satisfy: s_i <= 2*dx - s_j OR s_i <= 2*dy - s_j
        # The maximum s_i allowed by this pair is max(2*dx - s_j, 2*dy - s_j)
        pair_limit = max(2.0 * dx - sides[j], 2.0 * dy - sides[j])
        limit = min(limit, pair_limit)
    
    return max(0.0, limit)


def _voronoi_objective(x, tol=1e-7):
    """
    Objective: maximize sum of Voronoi-limited side lengths.
    x is a flat vector: [cx_0, cy_0, cx_1, cy_1, ..., cx_9, cy_9]
    """
    n = 10
    centers = x.reshape(n, 2)
    total = 0.0
    for i in range(n):
        cx, cy = centers[i]
        # Boundary limits
        max_from_boundary = min(cx, 1.0 - cx, cy, 1.0 - cy) * 2.0
        
        for j in range(n):
            if j == i:
                continue
            dx = abs(cx - centers[j, 0])
            dy = abs(cy - centers[j, 1])
            # For non-overlap: s_i <= 2*dx - s_j OR s_i <= 2*dy - s_j
            # But s_j itself is also variable. We use the current x to estimate s_j.
            # Actually, for the objective, we need a consistent definition.
            # Let's define s_j as the Voronoi-limited side for j given current positions.
            # This creates a coupled system. Instead, use a simpler proxy:
            # s_i_max = min over j of max(2*dx_ij - s_j_estimate, 2*dy_ij - s_j_estimate)
            # Use s_j_estimate = 0.25 as a reasonable average
            pass
        
        # Simpler: use the actual constraint structure
        # s_i is bounded by: for each j, max(2|dx_ij| - s_j, 2|dy_ij| - s_j)
        # Since s_j is also a function of positions, this is complex.
        # Alternative: define the objective as sum of min distances to boundaries/neighbors
        # A square of side s at (cx,cy) needs:
        #   cx - s/2 >= 0, cx + s/2 <= 1, etc.
        #   For each j: NOT(cx - s_i/2 < cx_j + s_j/2 AND cx_j - s_j/2 < cx + s_i/2 AND ...)
        # The max s_i given positions is what _voronoi_limited_side computes if s_j are fixed.
        # For the optimization, let's use a different approach:
        # Objective = sum_i of (distance to nearest constraint)
        
        # Distance to boundary
        d_bound = min(cx, 1.0 - cx, cy, 1.0 - cy)
        
        # Distance to each other square (Chebyshev-like for axis-aligned)
        # The gap between square i (side s_i) and square j (side s_j) in x is:
        #   gap_x = |cx_i - cx_j| - (s_i + s_j)/2
        # For non-overlap: gap_x >= 0 OR gap_y >= 0
        # The "free" space for s_i is limited by the min gap to any neighbor in the constraining direction.
        
        # Simplified objective: sum of min(d_bound_i, min_j gap_to_j)
        # where gap_to_j = max(|dx_ij|, |dy_ij|) - 0.25 (approximate)
        
        total += d_bound
    
    return -total  # Negate for minimization


def _slsqp_packing(centers, sides, tol=1e-7, max_iters=500, time_limit=30.0):
    """
    Use SLSQP to optimize positions and sizes simultaneously.
    Variables: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
    """
    n = 10
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[i*3] = centers[i, 0]
        x0[i*3+1] = centers[i, 1]
        x0[i*3+2] = sides[i]
    
    start_time = time.time()
    
    def unpack(x):
        c = np.zeros((n, 2))
        s = np.zeros(n)
        for i in range(n):
            c[i, 0] = x[i*3]
            c[i, 1] = x[i*3+1]
            s[i] = x[i*3+2]
        return c, s
    
    def objective(x):
        c, s = unpack(x)
        # Penalty for invalid configurations
        penalty = 0.0
        for i in range(n):
            if s[i] <= 0:
                penalty += 100.0 * (0.01 - s[i])
            half = s[i] / 2.0
            if c[i,0] - half < -tol:
                penalty += 100.0 * (-tol - (c[i,0] - half))
            if c[i,0] + half > 1.0 + tol:
                penalty += 100.0 * ((c[i,0] + half) - (1.0 + tol))
            if c[i,1] - half < -tol:
                penalty += 100.0 * (-tol - (c[i,1] - half))
            if c[i,1] + half > 1.0 + tol:
                penalty += 100.0 * ((c[i,1] + half) - (1.0 + tol))
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                dx = abs(c[i,0] - c[j,0])
                dy = abs(c[i,1] - c[j,1])
                if dx < t and dy < t:
                    # Overlap penalty
                    overlap_x = t - dx
                    overlap_y = t - dy
                    penalty += 100.0 * (overlap_x * overlap_x + overlap_y * overlap_y)
        return -np.sum(s) + penalty
    
    def gradient(x):
        c, s = unpack(x)
        grad = np.zeros(n * 3)
        for i in range(n):
            grad[i*3] = 0.0
            grad[i*3+1] = 0.0
            grad[i*3+2] = -1.0
        return grad
    
    # Constraints: non-overlap (as inequalities)
    # g(x) >= 0 where g = min(dx - t, dy - t) for each pair
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            def make_constraint(i, j):
                def constraint(x):
                    c, s = unpack(x)
                    t = (s[i] + s[j]) / 2.0 - tol
                    dx = abs(c[i,0] - c[j,0])
                    dy = abs(c[i,1] - c[j,1])
                    return min(dx - t, dy - t)
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.01, 1.0)) # s
    
    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iters, 'ftol': 1e-12, 'disp': False}
    )
    
    elapsed = time.time() - start_time
    if result.success and result.fun is not None:
        c, s = unpack(result.x)
        if _is_valid(c, s, tol):
            return c, s, elapsed
    return None, None, elapsed


def _smooth_voronoi_optimize(centers, sides, tol=1e-7, max_iters=300, lr=0.01, time_limit=20.0):
    """
    Gradient-based optimization where each square's side is the Voronoi-limited maximum.
    We optimize positions to maximize sum of these limited sides.
    """
    n = 10
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Voronoi-limited sides
        v_sides = np.zeros(n)
        for i in range(n):
            v_sides[i] = _voronoi_limited_side(centers, sides, i, tol)
        
        # Compute gradient of sum(v_sides) w.r.t. positions
        grad = np.zeros((n, 2))
        for i in range(n):
            cx, cy = centers[i]
            # d(v_sides[i])/d(cx) and d(v_sides[i])/d(cy)
            # v_sides[i] = min over constraints
            # For boundary: min(cx, 1-cx, cy, 1-cy) * 2
            # For neighbor j: max(2|dx_ij| - s_j, 2|dy_ij| - s_j)
            
            # Boundary contribution
            d_bound = min(cx, 1.0 - cx, cy, 1.0 - cy)
            g_cx = 2.0 if cx == d_bound else (-2.0 if (1.0 - cx) == d_bound else 0.0)
            g_cy = 2.0 if cy == d_bound else (-2.0 if (1.0 - cy) == d_bound else 0.0)
            
            # If boundary is the limiting factor
            limit_val = d_bound * 2.0
            is_boundary_limit = True
            for j in range(n):
                if j == i:
                    continue
                dx = abs(cx - centers[j, 0])
                dy = abs(cy - centers[j, 1])
                pair_limit = max(2.0 * dx - sides[j], 2.0 * dy - sides[j])
                if pair_limit < limit_val - 1e-12:
                    is_boundary_limit = False
                    # This neighbor is the limiting factor
                    if 2.0 * dx - sides[j] >= 2.0 * dy - sides[j]:
                        # x-direction is limiting
                        sign = 1.0 if cx >= centers[j, 0] else -1.0
                        g_cx = 2.0 * sign
                        g_cy = 0.0
                    else:
                        sign = 1.0 if cy >= centers[j, 1] else -1.0
                        g_cx = 0.0
                        g_cy = 2.0 * sign
                    limit_val = pair_limit
                    break
            
            if is_boundary_limit:
                if cx <= 1.0 - cx and cx <= cy and cx <= 1.0 - cy:
                    g_cx, g_cy = 2.0, 0.0
                elif 1.0 - cx <= cx and 1.0 - cx <= cy and 1.0 - cx <= 1.0 - cy:
                    g_cx, g_cy = -2.0, 0.0
                elif cy <= cx and cy <= 1.0 - cx and cy <= 1.0 - cy:
                    g_cx, g_cy = 0.0, 2.0
                else:
                    g_cx, g_cy = 0.0, -2.0
            
            grad[i, 0] += g_cx
            grad[i, 1] += g_cy
        
        # Update positions (gradient ascent on sum of v_sides)
        for i in range(n):
            centers[i, 0] += lr * grad[i, 0]
            centers[i, 1] += lr * grad[i, 1]
        
        # Clamp
        for i in range(n):
            half = sides[i] / 2.0
            centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
            centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
        
        # Grow sides to Voronoi limits
        for i in range(n):
            v = _voronoi_limited_side(centers, sides, i, tol)
            if v > sides[i]:
                sides[i] = v
        
        # Check validity
        if _is_valid(centers, sides, tol):
            obj = _objective(centers, sides)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = centers.copy()
                best_s = sides.copy()
    
    return best_c, best_s


def _high_precision_polish(centers, sides, tol=1e-7):
    """Final high-precision polish using tighter tolerances."""
    for _ in range(10):
        improved = False
        for i in range(10):
            ns = _grow_square_max(centers, sides, i, tol=1e-9)
            if ns > sides[i] + 1e-12:
                sides[i] = ns
                improved = True
        if not improved:
            break
    if _is_valid(centers, sides, tol):
        return centers, sides
    return centers, sides


def _init_grid_packing():
    """Baseline 3x3 grid packing (9 squares of 1/3 + 1 extra)."""
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[:9] = np.stack([cx, cy], axis=1)
    sides[:9] = side
    centers[9] = [0.5, 0.5]
    sides[9] = 1.0/6.0
    return centers, sides


def _init_heterogeneous_1():
    """One large (0.5) bottom-left, 9 smaller filling rest."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_2():
    """Two large (0.4) and 8 small (0.25)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    side_s = 0.25
    idx = 2
    for row in range(2):
        for col in range(4):
            centers[idx, 0] = (col + 0.5) * side_s
            centers[idx, 1] = 0.4 + (row + 0.5) * side_s
            sides[idx] = side_s
            idx += 1
    return centers, sides


def _init_heterogeneous_3():
    """One very large (0.5) + 9 small in L-shape."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """1 large (0.5) + 2 medium (0.25) + 7 small (0.125)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    side_s = 0.125
    idx = 3
    positions = [
        [0.375, 0.625], [0.625, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.75, 0.25],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    """Two large (0.45) in opposite corners, 8 smaller filling rest."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.775, 0.775]
    sides[1] = 0.45
    side_s = 0.15
    idx = 2
    positions = [
        [0.225, 0.575], [0.225, 0.875],
        [0.525, 0.225], [0.825, 0.225],
        [0.525, 0.525],
        [0.05, 0.575], [0.05, 0.875],
        [0.825, 0.525],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    """Three medium (0.33) in corners + 7 small."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.165, 0.165]
    sides[0] = 0.33
    centers[1] = [0.835, 0.165]
    sides[1] = 0.33
    centers[2] = [0.165, 0.835]
    sides[2] = 0.33
    side_s = 0.15
    idx = 3
    positions = [
        [0.5, 0.165], [0.835, 0.5], [0.5, 0.5],
        [0.5, 0.835], [0.835, 0.835], [0.33, 0.5],
        [0.665, 0.33],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_7():
    """One large (0.4) bottom-left, one large (0.35) top-right, rest fill."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.825, 0.825]
    sides[1] = 0.35
    side_s = 0.15
    idx = 2
    positions = [
        [0.55, 0.2], [0.8, 0.2],
        [0.2, 0.55], [0.2, 0.8],
        [0.55, 0.5], [0.55, 0.8],
        [0.2, 0.35], [0.55, 0.35],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_random(seed=123):
    """Random initial configuration."""
    rng = np.random.default_rng(seed=seed)
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    for i in range(10):
        s = rng.uniform(0.1, 0.35)
        sides[i] = s
        centers[i, 0] = rng.uniform(s/2, 1.0 - s/2)
        centers[i, 1] = rng.uniform(s/2, 1.0 - s/2)
    return centers, sides


def _resolve_overlaps(centers, sides, tol=1e-7):
    """Resolve overlaps using force-based separation."""
    return _force_based_separation(centers, sides, tol, max_iters=500)


def _local_optimize(centers, sides, tol=1e-7, max_rounds=40, seed=42, time_limit=60.0):
    """
    Enhanced local optimization combining smooth optimization, growth, and perturbations.
    """
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    rng = np.random.default_rng(seed=seed)
    start_time = time.time()
    
    for rnd in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        # Step 1: Grow all squares iteratively
        new_sides = _grow_all_iteratively(centers.copy(), sides.copy(), tol)
        if _is_valid(centers, new_sides, tol):
            obj = _objective(centers, new_sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = new_sides.copy()
                best_centers = centers.copy()
                sides = new_sides.copy()
        
        # Step 2: Smooth Voronoi optimization
        if rnd % 3 == 0:
            sc, ss = _smooth_voronoi_optimize(centers.copy(), sides.copy(), tol, max_iters=200, lr=0.01, time_limit=15.0)
            if _is_valid(sc, ss, tol):
                obj = _objective(sc, ss)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_sides = ss.copy()
                    best_centers = sc.copy()
                    centers = sc.copy()
                    sides = ss.copy()
        
        # Step 3: Force-based separation + regrow
        if rnd % 4 == 0:
            sep_centers = _force_based_separation(centers.copy(), sides, tol)
            if _is_valid(sep_centers, sides, tol):
                sep_sides = _grow_all_iteratively(sep_centers.copy(), sides.copy(), tol)
                if _is_valid(sep_centers, sep_sides, tol):
                    obj = _objective(sep_centers, sep_sides)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_sides = sep_sides.copy()
                        best_centers = sep_centers.copy()
                        centers = sep_centers.copy()
                        sides = sep_sides.copy()
        
        # Step 4: Small targeted perturbations
        for i in range(n):
            for _ in range(2):
                if time.time() - start_time > time_limit:
                    break
                dx = rng.uniform(-0.03, 0.03)
                dy = rng.uniform(-0.03, 0.03)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i]/2, 1.0 - sides[i]/2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i]/2, 1.0 - sides[i]/2)
                if _is_valid(new_c, sides, tol):
                    trial_s = _grow_all_iteratively(new_c.copy(), sides.copy(), tol)
                    if _is_valid(new_c, trial_s, tol):
                        obj = _objective(new_c, trial_s)
                        if obj > best_obj + 1e-9:
                            best_obj = obj
                            best_centers = new_c.copy()
                            best_sides = trial_s.copy()
                            centers = new_c.copy()
                            sides = trial_s.copy()
    
    return best_centers, best_sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s
    
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
    ]
    
    # Add random inits
    for s in [123, 456, 789, 1011, 2020, 3030]:
        inits.append(_init_random(seed=s))
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        
        # Resolve overlaps
        centers = _resolve_overlaps(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            centers = _force_based_separation(centers, sides, max_iters=1000)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
        
        # Local optimization with unique seed per init
        remaining = deadline - time.time()
        opt_time = min(45.0, max(10.0, remaining / 3.0))
        opt_centers, opt_sides = _local_optimize(centers, sides, seed=42 + idx * 100, time_limit=opt_time)
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # If we have a good result, try SLSQP on it
        if best_obj > 1.1 and time.time() < deadline - 60:
            sl_c, sl_s, elapsed = _slsqp_packing(best_centers.copy(), best_sides.copy(), tol=1e-7, max_iters=200, time_limit=40.0)
            if sl_c is not None and _is_valid(sl_c, sl_s):
                obj = _objective(sl_c, sl_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_centers = sl_c.copy()
                    best_sides = sl_s.copy()
        
        # Shake and regrow
        if best_obj > 1.1 and time.time() < deadline - 40:
            rng = np.random.default_rng(seed=999 + idx)
            for _ in range(3):
                if time.time() > deadline - 30:
                    break
                new_centers = best_centers.copy()
                for i in range(10):
                    dx = rng.uniform(-0.04, 0.04)
                    dy = rng.uniform(-0.04, 0.04)
                    new_centers[i, 0] = np.clip(best_centers[i, 0] + dx, best_sides[i]/2, 1.0 - best_sides[i]/2)
                    new_centers[i, 1] = np.clip(best_centers[i, 1] + dy, best_sides[i]/2, 1.0 - best_sides[i]/2)
                new_centers = _force_based_separation(new_centers, best_sides, max_iters=200)
                if _is_valid(new_centers, best_sides):
                    new_sides = _grow_all_iteratively(new_centers.copy(), best_sides.copy())
                    if _is_valid(new_centers, new_sides):
                        obj = _objective(new_centers, new_sides)
                        if obj > best_obj + 1e-9:
                            best_obj = obj
                            best_centers = new_centers.copy()
                            best_sides = new_sides.copy()
    
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final high-precision polish
        if time.time() < deadline - 10:
            best_centers, best_sides = _high_precision_polish(best_centers, best_sides)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
