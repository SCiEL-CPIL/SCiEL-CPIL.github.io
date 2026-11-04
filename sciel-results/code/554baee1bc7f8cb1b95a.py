"""Smooth constrained optimization for circle packing via L-BFGS-B."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _smooth_objective(params, n):
    """
    Smooth objective: minimize (penalty - sum_radii).
    params: array of length 3n = [x0, y0, r0, x1, y1, r1, ...]
    Penalty uses smooth max(0, violation)^2 for overlaps and wall violations.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Wall penalties: circle must be inside [0,1]^2
    # Violation if x - r < 0 => r - x > 0, etc.
    wall_viol = np.maximum(0.0, r - x) + np.maximum(0.0, r - y) + \
                np.maximum(0.0, x + r - 1.0) + np.maximum(0.0, y + r - 1.0)
    wall_pen = np.sum(wall_viol ** 2)
    
    # Pairwise overlap penalties
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            # Smooth overlap: max(0, r_i + r_j - dist)^2
            viol = r[i] + r[j] - dist
            if viol > 0:
                overlap_pen += viol * viol
    
    # Objective: we want to maximize sum(r), so minimize (penalty - sum(r))
    # But we also want radii to be positive and not too large
    obj = wall_pen + overlap_pen - np.sum(r)
    
    return obj


def _smooth_gradient(params, n):
    """Gradient of the smooth objective (for L-BFGS-B)."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Wall gradients
    # d/dx of max(0, r-x)^2 = -2*max(0, r-x)
    # d/dx of max(0, x+r-1)^2 = 2*max(0, x+r-1)
    # d/dr of max(0, r-x)^2 = 2*max(0, r-x)
    # d/dr of max(0, x+r-1)^2 = 2*max(0, x+r-1)
    
    v1 = np.maximum(0.0, r - x)
    v2 = np.maximum(0.0, r - y)
    v3 = np.maximum(0.0, x + r - 1.0)
    v4 = np.maximum(0.0, y + r - 1.0)
    
    gx -= 2.0 * v1
    gx += 2.0 * v3
    gy -= 2.0 * v2
    gy += 2.0 * v4
    gr += 2.0 * v1 + 2.0 * v2 + 2.0 * v3 + 2.0 * v4
    
    # Pairwise overlap gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            viol = r[i] + r[j] - dist
            if viol > 0:
                # d/dx_i of (r_i+r_j-dist)^2 = 2*viol * d/dx_i(-dist) = 2*viol * dx/dist
                # d/dx_j = -2*viol * dx/dist
                # d/dr_i = 2*viol, d/dr_j = 2*viol
                factor = 2.0 * viol / dist
                gx[i] += factor * dx
                gy[i] += factor * dy
                gx[j] -= factor * dx
                gy[j] -= factor * dy
                gr[i] += 2.0 * viol
                gr[j] += 2.0 * viol
    
    # Gradient of -sum(r)
    gr -= 1.0
    
    # Interleave
    grad = np.empty(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    return grad


def _optimize_from_init(x0, y0, r0, deadline, max_iter=2000):
    """Run L-BFGS-B optimization from given initial positions and radii."""
    n = len(x0)
    params = np.empty(3 * n)
    params[0::3] = x0
    params[1::3] = y0
    params[2::3] = r0
    
    # Bounds: positions in [0,1], radii in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    best_params = params.copy()
    best_obj = _smooth_objective(params, n)
    
    # Run L-BFGS-B
    result = minimize(
        _smooth_objective,
        params,
        jac=_smooth_gradient,
        args=(n,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if time.time() > deadline:
        # Return best found so far
        pass
    
    if result.fun < best_obj:
        best_obj = result.fun
        best_params = result.x
    
    # Extract and validate
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    # Clamp to ensure validity
    for i in range(n):
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        r[i] = max(r[i], 0.001)
        x[i] = np.clip(x[i], r[i], 1.0 - r[i])
        y[i] = np.clip(y[i], r[i], 1.0 - y[i])
    
    total, valid = _eval_packing(np.column_stack([x, y]), r)
    if not valid:
        # Try to repair: shrink radii slightly
        for i in range(n):
            r[i] *= 0.95
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - y[i])
        total, valid = _eval_packing(np.column_stack([x, y]), r)
        if not valid:
            return None, 0.0
    
    return np.column_stack([x, y]), r, total


def _expand_radii(xy, r, deadline, n_expand_steps=50):
    """
    Try to uniformly expand all radii while maintaining validity.
    This is a post-optimization polish: after finding valid positions,
    try to grow radii by small amounts and re-optimize positions.
    """
    n = len(xy)
    x = xy[:, 0].copy()
    y = xy[:, 1].copy()
    r = r.copy()
    
    best_total = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for step in range(n_expand_steps):
        if time.time() > deadline:
            break
        
        # Try to increase all radii by a small amount
        dr = 0.001
        r_new = r + dr
        
        # Check if this is feasible (all circles still inside and disjoint)
        feasible = True
        for i in range(n):
            if r_new[i] > min(x[i], 1.0 - x[i], y[i], 1.0 - y[i]):
                feasible = False
                break
        if feasible:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r_new[i] + r_new[j] - 1e-9:
                        feasible = False
                        break
                if not feasible:
                    break
        
        if feasible:
            r = r_new
            total = np.sum(r)
            if total > best_total:
                best_total = total
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        else:
            # Can't expand uniformly anymore; try local re-optimization
            # with slightly larger radii to see if positions can adjust
            break
    
    return best_x, best_y, best_r, best_total


def _hex_init():
    """Initialize circles in a hexagonal pattern."""
    positions = []
    side = int(np.ceil(np.sqrt(N)))
    spacing = 1.0 / (side + 0.5)
    r_init = spacing * 0.40
    
    row = 0
    while len(positions) < N:
        for col in range(side + 1):
            if len(positions) >= N:
                break
            x = (col + 0.25) * spacing + (0.25 if row % 2 == 1 else 0)
            y = (row + 0.5) * spacing
            if 0 < x < 1 and 0 < y < 1:
                positions.append([x, y, r_init])
        row += 1
    
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    
    return np.array(positions[:N])


def _random_init(seed=42):
    """Initialize circles at random positions."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(N):
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = 0.04
        positions.append([x, y, r])
    return np.array(positions)


def _grid_init(side=5):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.35 / side
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _staggered_init():
    """Staggered grid with varying radii."""
    positions = []
    rows = 5
    cols = 6
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= N:
                break
            x = (j + 0.5) / cols + (0.5 / cols if i % 2 == 1 else 0)
            y = (i + 0.5) / rows
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = 0.30 / cols
            positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_total = 0.0
    best_result = None

    initializations = [
        ("hex", _hex_init()),
        ("random42", _random_init(42)),
        ("random123", _random_init(123)),
        ("random777", _random_init(777)),
        ("grid5", _grid_init(5)),
        ("grid6", _grid_init(6)),
        ("staggered", _staggered_init()),
    ]

    for name, init_pos in initializations:
        if time.time() > deadline:
            break
        
        x = init_pos[:, 0].copy()
        y = init_pos[:, 1].copy()
        r = init_pos[:, 2].copy()
        
        # Run L-BFGS-B optimization
        xy_out, r_out, total = _optimize_from_init(x, y, r, deadline, max_iter=1500)
        
        if xy_out is not None and total > best_total:
            best_total = total
            best_result = np.column_stack([xy_out, r_out])
    
    # If no valid result, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.35 / side
        return result

    # Final polish: try to expand radii
    if time.time() < deadline - 20:
        xy = best_result[:, :2].copy()
        r = best_result[:, 2].copy()
        px, py, pr, pt = _expand_radii(xy, r, deadline)
        if pt > best_total:
            best_total = pt
            best_result = np.column_stack([np.column_stack([px, py]), pr])
    
    # Final validation and cleanup
    result = best_result.copy()
    n = len(result)
    for i in range(n):
        x, y, ri = result[i, 0], result[i, 1], result[i, 2]
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
            result[i, 2] = 0.35 / side

    return result
# EVOLVE-BLOCK-END
