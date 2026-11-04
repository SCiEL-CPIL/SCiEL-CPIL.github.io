"""
Strategy: Smooth Penalty Optimization with L-BFGS-B for Circle Packing

This approach uses a smooth quadratic penalty for overlaps and wall violations.
It iteratively optimizes positions and radii using L-BFGS-B, which is faster
and more robust than SLSQP for this problem size. It employs a growth strategy
to find larger packings.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _check_validity(circles, w, h, tol=1e-6):
    """Check if circles are valid: within bounds and non-overlapping."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > w + tol:
            return False
        if y - r < -tol or y + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True

def _separate_circles(circles, w, h, max_iter=500, tol=1e-8):
    """
    Repair invalid circles by iteratively separating overlapping pairs
    and pushing circles inside bounds.
    Returns (repaired_circles, valid).
    """
    circles = circles.copy()
    n = len(circles)
    
    for _ in range(max_iter):
        moved = False
        
        # Push circles inside bounds
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            elif x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            elif y + r > h:
                circles[i, 1] = h - r
                moved = True
        
        # Separate overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist:
                    if dist < 1e-12:
                        # Circles at same position; push apart
                        circles[i, 0] += min_dist / 2
                        circles[j, 0] -= min_dist / 2
                        moved = True
                    else:
                        overlap = min_dist - dist
                        # Push apart along the connecting line
                        dx_unit = dx / dist
                        dy_unit = dy / dist
                        circles[i, 0] += overlap / 2 * dx_unit
                        circles[i, 1] += overlap / 2 * dy_unit
                        circles[j, 0] -= overlap / 2 * dx_unit
                        circles[j, 1] -= overlap / 2 * dy_unit
                        moved = True
        
        # Re-check bounds after separation
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            elif x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            elif y + r > h:
                circles[i, 1] = h - r
                moved = True
        
        if not moved:
            break
    
    if _check_validity(circles, w, h, tol=1e-6):
        return circles, True
    else:
        # Final attempt: shrink radii slightly
        for scale in [0.995, 0.99, 0.98, 0.97, 0.95]:
            circles_shrunk = circles.copy()
            circles_shrunk[:, 2] *= scale
            if _check_validity(circles_shrunk, w, h, tol=1e-6):
                return circles_shrunk, True
        return circles, False

def _objective_and_grad(x, w, h, beta=100.0):
    """
    Objective function for optimization: minimize -sum(r_i) + penalty for violations.
    x = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = len(x) // 3
    x_coords = x[0::3]
    y_coords = x[1::3]
    r = x[2::3]
    
    # Base objective: maximize sum of radii
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Penalty for overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x_coords[i] - x_coords[j]
            dy = y_coords[i] - y_coords[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            
            if dist < min_dist:
                overlap = min_dist - dist
                # Smooth quadratic penalty
                penalty = 0.5 * beta * overlap * overlap
                obj += penalty
                
                if dist > 1e-12:
                    dx_unit = dx / dist
                    dy_unit = dy / dist
                    # Gradient w.r.t. x_i, y_i
                    grad[3 * i] += beta * overlap * dx_unit
                    grad[3 * i + 1] += beta * overlap * dy_unit
                    # Gradient w.r.t. x_j, y_j
                    grad[3 * j] -= beta * overlap * dx_unit
                    grad[3 * j + 1] -= beta * overlap * dy_unit
                    # Gradient w.r.t. r_i, r_j
                    grad[3 * i + 2] += beta * overlap
                    grad[3 * j + 2] += beta * overlap
                else:
                    # Degenerate: push apart in x
                    grad[3 * i] += beta * overlap
                    grad[3 * j] -= beta * overlap
                    grad[3 * i + 2] += beta * overlap
                    grad[3 * j + 2] += beta * overlap
    
    # Penalty for wall violations
    for i in range(n):
        # Left wall
        viol = max(0, r[i] - x_coords[i])
        if viol > 0:
            penalty = 0.5 * beta * viol * viol
            obj += penalty
            grad[3 * i] -= beta * viol  # x_i
            grad[3 * i + 2] += beta * viol  # r_i
        
        # Right wall
        viol = max(0, x_coords[i] + r[i] - w)
        if viol > 0:
            penalty = 0.5 * beta * viol * viol
            obj += penalty
            grad[3 * i] += beta * viol  # x_i
            grad[3 * i + 2] += beta * viol  # r_i
        
        # Bottom wall
        viol = max(0, r[i] - y_coords[i])
        if viol > 0:
            penalty = 0.5 * beta * viol * viol
            obj += penalty
            grad[3 * i + 1] -= beta * viol  # y_i
            grad[3 * i + 2] += beta * viol  # r_i
        
        # Top wall
        viol = max(0, y_coords[i] + r[i] - h)
        if viol > 0:
            penalty = 0.5 * beta * viol * viol
            obj += penalty
            grad[3 * i + 1] += beta * viol  # y_i
            grad[3 * i + 2] += beta * viol  # r_i
    
    return obj, grad

def _optimize_pack(circles, w, h, max_iter=300, beta_start=100.0, beta_end=1000.0):
    """
    Optimize circle positions and radii using L-BFGS-B with increasing penalty.
    """
    x0 = circles.flatten()
    n = len(circles)
    
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, min(w, h) * 0.5)) # Max radius bound
    
    # Schedule beta increase
    betas = np.linspace(beta_start, beta_end, 5)
    x_current = x0
    
    for beta in betas:
        res = minimize(
            lambda x: _objective_and_grad(x, w, h, beta=beta)[0],
            x_current,
            jac=lambda x: _objective_and_grad(x, w, h, beta=beta)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter // len(betas), 'ftol': 1e-12, 'gtol': 1e-8}
        )
        x_current = res.x
        
    return x_current.reshape(n, 3)

def _grid_initialization(w, h, n_circles):
    """Create an initial grid placement of circles."""
    side = int(np.ceil(np.sqrt(n_circles)))
    circles = np.zeros((n_circles, 3))
    for i in range(n_circles):
        row = i // side
        col = i % side
        circles[i, 0] = (col + 0.5) * (w / side)
        circles[i, 1] = (row + 0.5) * (h / side)
        min_dim = min(w, h)
        circles[i, 2] = 0.5 * min_dim / side * 0.3  # Small initial radius
    return circles

def _random_initialization(w, h, n_circles, seed=42):
    """Create a random initial placement of circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n_circles, 3))
    for i in range(n_circles):
        circles[i, 0] = rng.uniform(0.05, w - 0.05)
        circles[i, 1] = rng.uniform(0.05, h - 0.05)
        circles[i, 2] = rng.uniform(0.02, 0.08)
    return circles

def _best_known():
    """
    Return a known good packing for 21 circles in a 1x1 square (perimeter 4).
    This is a fallback; the actual optimization should do better.
    """
    # Simple 5x5 grid with one extra circle
    w, h = 1.0, 1.0
    side = 5
    circles = np.zeros((N, 3))
    r = 0.08
    for i in range(N):
        row = i // side
        col = i % side
        circles[i, 0] = (col + 0.5) / side
        circles[i, 1] = (row + 0.5) / side
        circles[i, 2] = r
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    # Broader aspect ratio search
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.25, 1.1, 0.9, 1.4, 0.7, 1.6, 1.8, 0.55, 1.45]
    
    # Shuffle to ensure we try diverse aspects early if time is tight
    rng = np.random.RandomState(42)
    rng.shuffle(aspect_ratios)
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Initial grid placement
        circles = _grid_initialization(w, h, N)
        
        # Optimization
        circles_opt = _optimize_pack(circles, w, h, max_iter=200, beta_start=50.0, beta_end=500.0)
        
        # Check validity and repair if needed
        if not _check_validity(circles_opt, w, h, tol=1e-5):
            circles_fixed, valid = _separate_circles(circles_opt, w, h, max_iter=100, tol=1e-6)
            if valid:
                circles_opt = circles_fixed
            else:
                continue
        
        value = np.sum(circles_opt[:, 2])
        if value > best_value:
            best_value = value
            best_result = circles_opt.copy()
            
    # If we still have time, try random restarts with the best aspect found or new ones
    if best_result is not None and time.time() < deadline - 20.0:
        # Determine the best aspect ratio from the best result
        # We don't store aspect, so we just try a few random restarts with square
        # Or we can infer aspect from best_result bounding box? No, we fixed w,h in loop.
        # Let's just do a few high-precision polish runs on the best result found.
        
        # Re-derive w, h for the best result is hard without storing them.
        # Instead, let's just run a few more random seeds for square and 1.2 aspect.
        for aspect in [1.0, 1.2]:
            if time.time() > deadline - 10.0:
                break
            w = 2.0 / (1.0 + aspect)
            h = 2.0 - w
            
            for seed in [1, 2, 3]:
                if time.time() > deadline - 5.0:
                    break
                circles = _random_initialization(w, h, N, seed=seed + 100)
                circles_opt = _optimize_pack(circles, w, h, max_iter=100, beta_start=50.0, beta_end=500.0)
                
                if not _check_validity(circles_opt, w, h, tol=1e-5):
                    circles_fixed, valid = _separate_circles(circles_opt, w, h, max_iter=100, tol=1e-6)
                    if valid:
                        circles_opt = circles_fixed
                    else:
                        continue
                
                value = np.sum(circles_opt[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = circles_opt.copy()

    if best_result is None:
        # Fallback
        best_result = _best_known()
    
    return best_result
