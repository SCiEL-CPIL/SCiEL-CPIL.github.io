"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _penalty(x, w, h, margin=1e-4):
    """
    x: array of shape (N, 3) with columns [cx, cy, r]
    w, h: rectangle dimensions
    Returns penalty value (lower is better)
    """
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    n = len(x)
    penalty = 0.0
    
    # Boundary constraints: circles must be inside the rectangle
    # Left: cx - r >= 0  => r - cx <= 0
    # Right: cx + r <= w => cx + r - w <= 0
    # Bottom: cy - r >= 0
    # Top: cy + r <= h
    left_viol = np.maximum(0, r - cx)
    right_viol = np.maximum(0, cx + r - w)
    bottom_viol = np.maximum(0, r - cy)
    top_viol = np.maximum(0, cy + r - h)
    penalty += 100 * (np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2))
    
    # Disjoint interiors: distance between centers >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                viol = min_dist - dist
                penalty += 100 * viol**2
    
    return penalty

def _objective(x, w, h):
    """
    Objective: minimize negative sum of radii + penalty
    """
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    # Ensure radii are positive
    r_clipped = np.maximum(r, 1e-6)
    neg_sum_r = -np.sum(r_clipped)
    pen = _penalty(x, w, h)
    return neg_sum_r + pen

def _initial_guess(w, h, n):
    """
    Create an initial grid-based guess
    """
    side = int(np.ceil(np.sqrt(n)))
    # Spread circles in a grid within the rectangle
    xs = np.linspace(0.5 / side * w, (side - 0.5) / side * w, side)
    ys = np.linspace(0.5 / side * h, (side - 0.5) / side * h, side)
    
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            points.append([xs[i], ys[j], min(w, h) / (2 * side) * 0.8])
        if len(points) >= n:
            break
    
    # If we didn't get enough, add more
    while len(points) < n:
        points.append([w / 2, h / 2, min(w, h) / (2 * side) * 0.8])
    
    return np.array(points[:n])

def _validate(x, w, h, tol=1e-4):
    """
    Check if the solution is valid: all circles inside rectangle and disjoint
    """
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    n = len(x)
    
    # Check boundaries
    if np.any(cx - r < -tol) or np.any(cx + r > w + tol) or np.any(cy - r < -tol) or np.any(cy + r > h + tol):
        return False
    
    # Check disjointness
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True

def _compute_value(x, w, h):
    """
    Compute the sum of radii if valid, else 0
    """
    if not _validate(x, w, h):
        return 0.0
    return np.sum(x[:, 2])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_value = 0.0
    best_x = None
    
    # Try different aspect ratios
    # Perimeter = 2*(w+h) = 4 => w+h = 2
    # Let w = w, h = 2 - w
    # w in (0, 2), h in (0, 2)
    
    aspect_ratios = [
        (1.0, 1.0),      # Square
        (1.2, 0.8),
        (0.8, 1.2),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.8, 0.2),
        (0.2, 1.8),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.3, 0.7),
        (0.7, 1.3),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Initial guess
        x0 = _initial_guess(w, h, N)
        
        # Bounds: cx in [r, w-r], cy in [r, h-r], r in [1e-4, min(w,h)/2]
        # But since r is variable, we need to handle this carefully
        # We'll use bounds that are generous and rely on penalties
        max_r = min(w, h) / 2
        bounds = [(1e-6, w - 1e-6), (1e-6, h - 1e-6), (1e-6, max_r)] * N
        
        # Optimize
        result = minimize(
            _objective,
            x0,
            args=(w, h),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x
        value = _compute_value(x_opt, w, h)
        
        if value > best_value:
            best_value = value
            best_x = x_opt.copy()
    
    # If we found a valid solution, return it
    if best_x is not None:
        # Final validation
        if _validate(best_x, *aspect_ratios[0]):
            return best_x
    
    # Fallback: return a simple valid packing
    # Use a grid in a square of side 1 (perimeter 4)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([cx, cy, r])
        if len(cells) >= N:
            break
    
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
