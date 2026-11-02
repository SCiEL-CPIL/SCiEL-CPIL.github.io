"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 21

# EVOLVE-BLOCK-START

def _construct_grid(n_circles, aspect):
    """
    Create a grid-based initial packing for n_circles in a rectangle with given aspect ratio.
    Returns (positions, radii, w, h)
    """
    w = 2.0 / (1.0 + aspect)
    h = 2.0 * aspect / (1.0 + aspect)
    
    # Determine grid dimensions
    # Try to find a grid that fits n_circles
    best_grid = (1, n_circles)
    best_score = 0
    for cols in range(1, n_circles + 1):
        rows = (n_circles + cols - 1) // cols
        # Score: prefer more square-like grids
        score = 1.0 / (1.0 + abs(cols - rows) * 0.1)
        if score > best_score:
            best_score = score
            best_grid = (cols, rows)
    
    cols, rows = best_grid
    
    positions = []
    radii = []
    
    for i in range(cols):
        for j in range(rows):
            if len(positions) >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            r = min(w / cols, h / rows) / 2.0 * 0.999
            positions.append([x, y])
            radii.append(r)
    
    # If we have more circles than needed, trim
    positions = positions[:n_circles]
    radii = radii[:n_circles]
    
    # If we have fewer, add more in a spiral or just duplicate positions with tiny radii
    while len(positions) < n_circles:
        positions.append([w / 2.0, h / 2.0])
        radii.append(0.01)
    
    return np.array(positions), np.array(radii), w, h


def _packing_loss(x, n_circles):
    """
    x = [x_1, y_1, r_1, ..., x_n, y_n, r_n, w, h]
    Returns (loss, grad) for SLSQP
    """
    n = n_circles
    w = x[-2]
    h = x[-1]
    
    # Perimeter constraint: 2*(w+h) <= 4, so w+h <= 2
    # We'll enforce this via bounds and constraints
    
    positions = x[:-2].reshape(-1, 3)
    xs = positions[:, 0]
    ys = positions[:, 1]
    rs = positions[:, 2]
    
    loss = 0.0
    grad = np.zeros_like(x)
    
    # Objective: maximize sum of radii (so minimize negative)
    # loss = -sum(rs)
    loss = -np.sum(rs)
    
    # Gradient of objective
    for i in range(n):
        grad[3*i + 2] = -1.0  # d(-r_i)/dr_i = -1
    
    # Constraints (penalty for violations):
    # 1. Circle i inside rectangle: 0 <= x_i - r_i, x_i + r_i <= w, 0 <= y_i - r_i, y_i + r_i <= h
    # 2. Circle i and j non-overlapping: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # Boundary constraints (soft penalty)
    for i in range(n):
        # x_i - r_i >= 0
        v1 = xs[i] - rs[i]
        if v1 < 0:
            loss += 100.0 * v1 * v1
            grad[3*i + 0] += -200.0 * v1
            grad[3*i + 2] += 200.0 * v1
        
        # x_i + r_i <= w
        v2 = w - (xs[i] + rs[i])
        if v2 < 0:
            loss += 100.0 * v2 * v2
            grad[-2] += 200.0 * v2
            grad[3*i + 0] += -200.0 * v2
            grad[3*i + 2] += -200.0 * v2
        
        # y_i - r_i >= 0
        v3 = ys[i] - rs[i]
        if v3 < 0:
            loss += 100.0 * v3 * v3
            grad[3*i + 1] += -200.0 * v3
            grad[3*i + 2] += 200.0 * v3
        
        # y_i + r_i <= h
        v4 = h - (ys[i] + rs[i])
        if v4 < 0:
            loss += 100.0 * v4 * v4
            grad[-1] += 200.0 * v4
            grad[3*i + 1] += -200.0 * v4
            grad[3*i + 2] += -200.0 * v4
    
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            rsum = rs[i] + rs[j]
            rsum_sq = rsum * rsum
            
            if dist_sq < rsum_sq:
                # Penalty
                violation = rsum_sq - dist_sq
                loss += 100.0 * violation * violation
                
                # Gradient
                dist = np.sqrt(max(dist_sq, 1e-12))
                factor = 200.0 * violation / dist
                
                grad[3*i + 0] += factor * dx
                grad[3*i + 1] += factor * dy
                grad[3*j + 0] -= factor * dx
                grad[3*j + 1] -= factor * dy
                grad[3*i + 2] += 200.0 * violation * rs[i] / rsum if rsum > 0 else 0
                grad[3*j + 2] += 200.0 * violation * rs[j] / rsum if rsum > 0 else 0
    
    return loss, grad


def _is_valid(positions, radii, w, h, tol=1e-6):
    """Check if packing is valid."""
    n = len(radii)
    
    # Perimeter
    if 2.0 * (w + h) > 4.0 + tol:
        return False
    
    # Boundary
    for i in range(n):
        if positions[i, 0] - radii[i] < -tol or positions[i, 0] + radii[i] > w + tol:
            return False
        if positions[i, 1] - radii[i] < -tol or positions[i, 1] + radii[i] > h + tol:
            return False
    
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            rsum = radii[i] + radii[j]
            if dist_sq < rsum*rsum - tol:
                return False
    
    return True


def _optimize_packing(initial_x, n_circles, max_iter=2000, tol=1e-8):
    """Run SLSQP optimization."""
    n = n_circles
    
    def objective(x):
        loss, _ = _packing_loss(x, n_circles)
        return loss
    
    def gradient(x):
        _, grad = _packing_loss(x, n_circles)
        return grad
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0))  # x_i
        bounds.append((1e-6, 1.0))  # y_i
        bounds.append((1e-6, 1.0))  # r_i
    bounds.append((0.1, 1.9))  # w
    bounds.append((0.1, 1.9))  # h
    
    # Constraints: w + h <= 2 (perimeter <= 4)
    def perimeter_constraint(x):
        return 2.0 - (x[-2] + x[-1])
    
    constraints = [{'type': 'ineq', 'fun': perimeter_constraint}]
    
    try:
        result = minimize(
            objective,
            initial_x,
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': tol, 'disp': False}
        )
        return result.x
    except Exception:
        return initial_x


def _refine_with_aspect(x_opt, n_circles, aspect_range=None):
    """Try different aspect ratios to improve the packing."""
    if aspect_range is None:
        aspect_range = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    
    best_x = x_opt.copy()
    best_loss = _packing_loss(x_opt, n_circles)[0]
    
    for aspect in aspect_range:
        w = 2.0 / (1.0 + aspect)
        h = 2.0 * aspect / (1.0 + aspect)
        
        # Scale positions to fit in this rectangle
        old_w = x_opt[-2]
        old_h = x_opt[-1]
        
        x_new = x_opt.copy()
        for i in range(n_circles):
            x_new[3*i + 0] = x_opt[3*i + 0] * w / old_w
            x_new[3*i + 1] = x_opt[3*i + 1] * h / old_h
            x_new[3*i + 2] = x_opt[3*i + 2] * min(w / old_w, h / old_h)
        x_new[-2] = w
        x_new[-1] = h
        
        result = _optimize_packing(x_new, n_circles, max_iter=1000)
        loss = _packing_loss(result, n_circles)[0]
        
        if loss < best_loss:
            best_loss = loss
            best_x = result
    
    return best_x


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 10s margin
    
    # Initial grid-based packing
    positions, radii, w, h = _construct_grid(N, aspect=1.0)
    
    # Build initial x vector
    x_init = np.zeros(3 * N + 2)
    for i in range(N):
        x_init[3*i + 0] = positions[i, 0]
        x_init[3*i + 1] = positions[i, 1]
        x_init[3*i + 2] = radii[i]
    x_init[-2] = w
    x_init[-1] = h
    
    # First optimization
    x_opt = _optimize_packing(x_init, N, max_iter=3000)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        # Not enough time for refinement
        pass
    else:
        # Try aspect ratio refinement
        x_opt = _refine_with_aspect(x_opt, N)
    
    # Extract result
    positions = x_opt[:-2].reshape(-1, 3)
    w = x_opt[-2]
    h = x_opt[-1]
    
    # Validate and fix
    result = np.zeros((N, 3))
    for i in range(N):
        x, y, r = positions[i]
        # Clamp to valid range
        r = max(1e-6, min(r, min(w, h) / 2.0 * 0.999))
        x = max(r, min(x, w - r))
        y = max(r, min(y, h - r))
        result[i] = [x, y, r]
    
    # Final validation
    if not _is_valid(result[:, :2], result[:, 2], w, h, tol=1e-4):
        # Fallback to grid
        positions, radii, w, h = _construct_grid(N, aspect=1.0)
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [positions[i, 0], positions[i, 1], radii[i]]
    
    return result

# EVOLVE-BLOCK-END
