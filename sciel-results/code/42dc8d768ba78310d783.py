"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Strategy:
- Treat rectangle aspect ratio (W+H=2) as a decision variable.
- Use a smooth, differentiable objective for scipy L-BFGS-B optimization.
- Initialize from hexagonal grids with multiple aspect ratios and seeds.
- Polish with iterative local optimization.
- Validate with strict geometric checks.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _check_valid(circles, tol=1e-7):
    """Check if circles are valid: disjoint, inside bounding rect with perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(rs < -1e-9):
        return False
    rs = np.maximum(rs, 0)
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + tol:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))


def _make_hex_grid(W, H, seed=0, n_attempts=50):
    """
    Create a hexagonal lattice initial configuration.
    Returns (N, 3) array of [x, y, r].
    """
    rng = np.random.RandomState(seed)
    
    # Try to find a radius that fits N circles in hexagonal packing
    # Hexagonal: horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # Odd rows offset by r
    
    best = None
    best_score = -np.inf
    
    for attempt in range(n_attempts):
        # Estimate radius from area
        r_est = np.sqrt(0.9069 * W * H / (N * np.pi))
        # Add some variation
        r = r_est * (0.7 + 0.5 * rng.rand())
        
        positions = []
        y = r
        row_idx = 0
        while y <= H - r + 1e-9:
            if row_idx % 2 == 0:
                x_start = r
            else:
                x_start = r + r  # offset by r
            x = x_start
            while x <= W - r + 1e-9:
                positions.append((x, y, r))
                x += 2.0 * r
            y += np.sqrt(3) * r
            row_idx += 1
        
        if len(positions) >= N:
            # Take first N positions
            pos = np.array(positions[:N])
            # Check if all are within bounds
            xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
            if np.all(xs - rs >= -1e-9) and np.all(xs + rs <= W + 1e-9) and \
               np.all(ys - rs >= -1e-9) and np.all(ys + rs <= H + 1e-9):
                # Check no overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if d < rs[i] + rs[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    score = np.sum(rs)
                    if score > best_score:
                        best_score = score
                        best = pos.copy()
        
        # Shrink for next attempt
        r_est *= 0.97
    
    if best is not None:
        return best
    
    # Fallback: square grid
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        cx = (col + 0.5) * W / side
        cy = (row + 0.5) * H / side
        r = 0.5 * min(W / side, H / side) * 0.9
        pos[i] = [cx, cy, r]
    return pos


def _smooth_objective(params, W, H, beta=50.0):
    """
    Smooth objective for optimization.
    params: flat array [x0,y0,r0, x1,y1,r1, ..., x20,y20,r20]
    Returns: -sum(r) + penalty (to minimize)
    """
    c = params.reshape(N, 3)
    xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
    
    # Ensure positive radii
    rs = np.maximum(rs, 1e-6)
    
    penalty = 0.0
    
    # Boundary penalties (smooth)
    # Left: x - r >= 0
    viol_l = np.maximum(0, rs - xs)
    viol_r = np.maximum(0, xs + rs - W)
    viol_b = np.maximum(0, rs - ys)
    viol_t = np.maximum(0, ys + rs - H)
    
    # Use smooth approximation: max(0, v) ~ v + sqrt(v^2 + eps) - sqrt(eps)
    eps = 1e-6
    def smooth_max(v):
        return v + np.sqrt(v*v + eps) - np.sqrt(eps)
    
    pen_boundary = (np.sum(smooth_max(viol_l)) + np.sum(smooth_max(viol_r)) +
                    np.sum(smooth_max(viol_b)) + np.sum(smooth_max(viol_t)))
    penalty += 100.0 * pen_boundary
    
    # Overlap penalties
    # For each pair, distance should be >= r_i + r_j
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = rs[i] + rs[j]
            viol = max(0, min_d - d)
            penalty += 50.0 * viol * viol
    
    obj = -np.sum(rs) + penalty
    return obj


def _gradient_objective(params, W, H, beta=50.0):
    """
    Objective with analytic gradient for faster convergence.
    """
    c = params.reshape(N, 3)
    xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
    rs_pos = np.maximum(rs, 1e-6)
    
    obj = -np.sum(rs_pos)
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0  # dr component
    
    # Boundary gradients
    eps = 1e-6
    for i in range(N):
        # Left: x_i - r_i >= 0
        v_l = rs_pos[i] - xs[i]
        if v_l > 0:
            s = np.sqrt(v_l*v_l + eps)
            grad[3*i] += 100.0 * (1.0 + v_l/s)  # dx
            grad[3*i+2] += 100.0 * (1.0 - v_l/s)  # dr (since v = r - x, dv/dr = 1, dv/dx = -1)
        
        # Right: x_i + r_i - W >= 0
        v_r = xs[i] + rs_pos[i] - W
        if v_r > 0:
            s = np.sqrt(v_r*v_r + eps)
            grad[3*i] += 100.0 * (1.0 + v_r/s)
            grad[3*i+2] += 100.0 * (1.0 + v_r/s)
        
        # Bottom: y_i - r_i >= 0
        v_b = rs_pos[i] - ys[i]
        if v_b > 0:
            s = np.sqrt(v_b*v_b + eps)
            grad[3*i+1] += 100.0 * (1.0 + v_b/s)
            grad[3*i+2] += 100.0 * (1.0 - v_b/s)
        
        # Top: y_i + r_i - H >= 0
        v_t = ys[i] + rs_pos[i] - H
        if v_t > 0:
            s = np.sqrt(v_t*v_t + eps)
            grad[3*i+1] += 100.0 * (1.0 + v_t/s)
            grad[3*i+2] += 100.0 * (1.0 + v_t/s)
    
    # Overlap gradients
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-10:
                d = 1e-10
                dx = 1e-10
                dy = 0
            min_d = rs_pos[i] + rs_pos[j]
            viol = min_d - d
            if viol > 0:
                # d(viol)/dx_i = -dx/d, d(viol)/dx_j = dx/d
                # d(viol)/dr_i = 1, d(viol)/dr_j = 1
                # penalty = 50 * viol^2
                # d(penalty)/dx_i = 100 * viol * (-dx/d)
                g = 100.0 * viol / d
                grad[3*i] += -g * dx
                grad[3*i+1] += -g * dy
                grad[3*i+2] += 100.0 * viol
                grad[3*j] += g * dx
                grad[3*j+1] += g * dy
                grad[3*j+2] += 100.0 * viol
    
    return obj, grad


def _optimize_from_init(circles, W, H, deadline, max_iter=500):
    """
    Optimize from initial configuration using L-BFGS-B.
    Returns optimized circles or original if failed.
    """
    params0 = circles.flatten()
    
    try:
        res = minimize(
            _gradient_objective,
            params0,
            args=(W, H),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        new_circles = res.x.reshape(N, 3)
        
        # Clamp to valid range
        for i in range(N):
            new_circles[i, 0] = np.clip(new_circles[i, 0], 0, W)
            new_circles[i, 1] = np.clip(new_circles[i, 1], 0, H)
            new_circles[i, 2] = max(new_circles[i, 2], 0)
        
        if _check_valid(new_circles):
            return new_circles
        else:
            return circles
    except Exception:
        return circles


def _try_aspect_ratio(W, H, deadline, seeds=[42, 123, 777, 2024]):
    """
    Try multiple initial configurations for a given aspect ratio.
    Returns best circles found or None.
    """
    best = None
    best_sum = -np.inf
    
    for seed in seeds:
        if time.time() > deadline - 2:
            break
        
        init = _make_hex_grid(W, H, seed=seed)
        
        # Optimize
        result = _optimize_from_init(init, W, H, deadline)
        
        if result is not None and _check_valid(result):
            s = _sum_radii(result)
            if s > best_sum:
                best_sum = s
                best = result.copy()
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_circles = None
    best_sum = -np.inf
    
    # Try various aspect ratios (W + H = 2)
    aspect_configs = [
        (1.0, 1.0),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.6, 0.4),
        (0.4, 1.6),
    ]
    
    for W, H in aspect_configs:
        if time.time() > deadline - 5:
            break
        
        result = _try_aspect_ratio(W, H, deadline)
        
        if result is not None and _check_valid(result):
            s = _sum_radii(result)
            if s > best_sum:
                best_sum = s
                best_circles = result.copy()
    
    # If we found a good solution, try to refine it further
    if best_circles is not None and time.time() < deadline - 3:
        # Get the bounding box of the best solution
        xs, ys, rs = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        W = max_x - min_x
        H = max_y - min_y
        
        # Shift to origin
        shifted = best_circles.copy()
        shifted[:, 0] -= min_x
        shifted[:, 1] -= min_y
        
        # Try to optimize in this tighter box
        refined = _optimize_from_init(shifted, W, H, deadline, max_iter=300)
        
        if refined is not None and _check_valid(refined):
            s = _sum_radii(refined)
            if s > best_sum:
                best_sum = s
                best_circles = refined.copy()
    
    # Final fallback
    if best_circles is None:
        # Simple grid in 1x1 square
        side = 5
        pos = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r = 0.5 / side - 0.01
            pos[i] = [cx, cy, r]
        best_circles = pos
    
    # Ensure non-negative radii
    best_circles[:, 2] = np.maximum(best_circles[:, 2], 0)
    
    return best_circles
