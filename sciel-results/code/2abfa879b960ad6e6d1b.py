"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _grid_init_staggered(n, w, h, cols, rows):
    """Create a staggered grid of n circles in rectangle w x h for better initial packing."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            # Stagger odd rows
            offset = 0.0 if i % 2 == 0 else 0.5 / cols
            cx = (j + 0.5 + offset) / cols * w
            cy = (i + 0.5) / rows * h
            # Estimate radius based on local spacing
            r = 0.5 / max(cols, rows) * min(w, h) * 0.95 - 1e-3
            r = max(r, 0.01)
            circles.append((cx, cy, r))
    
    circles = circles[:n]
    while len(circles) < n:
        circles.append((w / 2, h / 2, 0.01))
    return circles

def _smooth_objective(x, w, h, n, penalty_weight=100.0, smooth_eps=1e-4):
    """
    Smooth penalty-based objective for circle packing.
    Returns negative sum of radii minus penalties for constraint violations.
    Uses soft penalties to allow gradient-based methods to work effectively.
    """
    if time.time() > _deadline:
        return 0.0
    
    # Extract circles
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Sum of radii (we want to maximize this)
    total_r = np.sum(r)
    
    # Penalty for negative radii
    neg_r_penalty = np.sum(np.maximum(0, -r) ** 2) * penalty_weight
    
    # Penalty for circles outside rectangle (smooth)
    # Left: cx - r < 0
    left_viol = np.maximum(0, -(cx - r))
    # Right: cx + r > w
    right_viol = np.maximum(0, (cx + r) - w)
    # Bottom: cy - r < 0
    bottom_viol = np.maximum(0, -(cy - r))
    # Top: cy + r > h
    top_viol = np.maximum(0, (cy + r) - h)
    
    boundary_penalty = penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) + 
                                         np.sum(bottom_viol**2) + np.sum(top_viol**2))
    
    # Penalty for overlapping circles (smooth)
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            # Smooth penalty: penalize when dist < min_dist
            if dist_sq < (min_dist + smooth_eps) ** 2:
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 0.0
                violation = max(0, min_dist - dist)
                overlap_penalty += penalty_weight * violation ** 2
    
    return -(total_r) + neg_r_penalty + boundary_penalty + overlap_penalty

def _local_optimize_smooth(circles, w, h, deadline):
    """
    Local optimization using L-BFGS-B with smooth penalty objective.
    """
    global _deadline
    _deadline = deadline
    
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    # Use L-BFGS-B which handles the smooth penalty well
    try:
        res = minimize(
            _smooth_objective,
            x0,
            args=(w, h, n),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        # Check if result is valid
        best_circles = []
        for i in range(n):
            best_circles.append((res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]))
        
        if _validate(best_circles, w, h):
            return best_circles
    except Exception:
        pass
    
    # Fallback to Nelder-Mead if L-BFGS-B fails
    try:
        def nm_objective(x):
            if time.time() > deadline:
                return 0.0
            circles = [(x[i*3], x[i*3+1], x[i*3+2]) for i in range(n)]
            for cx, cy, r in circles:
                if r <= 0:
                    return 0.0
                if cx - r < -1e-9 or cx + r > w + 1e-9 or cy - r < -1e-9 or cy + r > h + 1e-9:
                    return 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < circles[i][2] + circles[j][2] - 1e-9:
                        return 0.0
            return -_total_radius(circles)
        
        res2 = minimize(
            nm_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        best_circles = [(res2.x[i*3], res2.x[i*3+1], res2.x[i*3+2]) for i in range(n)]
        if _validate(best_circles, w, h):
            return best_circles
    except Exception:
        pass
    
    return None

def _polish(circles, w, h, deadline):
    """High-precision polish pass with tighter tolerances."""
    if time.time() > deadline - 2.0:
        return circles
    
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    try:
        def polished_objective(x):
            if time.time() > deadline:
                return 0.0
            circles = [(x[i*3], x[i*3+1], x[i*3+2]) for i in range(n)]
            for cx, cy, r in circles:
                if r <= 0:
                    return 0.0
                if cx - r < -1e-10 or cx + r > w + 1e-10 or cy - r < -1e-10 or cy + r > h + 1e-10:
                    return 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < circles[i][2] + circles[j][2] - 1e-10:
                        return 0.0
            return -_total_radius(circles)
        
        res = minimize(
            polished_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        
        best_circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)]
        if _validate(best_circles, w, h, tol=1e-9):
            return best_circles
    except Exception:
        pass
    
    return circles

# Global deadline for smooth objective
_deadline = 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios to try
    aspect_ratios = [
        (1.0, 1.0),
        (0.85, 1.15),
        (0.75, 1.25),
        (0.7, 1.3),
        (0.9, 1.1),
        (0.8, 1.2),
    ]
    
    # Grid configurations
    grid_configs = [
        (5, 5),
        (4, 6),
        (6, 4),
        (5, 4),
        (4, 5),
        (3, 7),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline:
            break
        
        for cols, rows in grid_configs:
            if time.time() > deadline:
                break
            
            # Create staggered initial grid
            circles = _grid_init_staggered(N, w, h, cols, rows)
            
            if not _validate(circles, w, h):
                continue
            
            score = _total_radius(circles)
            if score > best_score:
                best_score = score
                best_circles = circles
            
            # Try smooth local optimization
            if time.time() < deadline - 10.0:
                optimized = _local_optimize_smooth(circles, w, h, deadline)
                if optimized is not None:
                    opt_score = _total_radius(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_circles = optimized
    
    # Final polish pass on the best result
    if best_circles is not None and time.time() < deadline - 3.0:
        # Find the best aspect ratio for the current best
        # Reconstruct w, h from the best circles
        max_x = max(c[0] + c[2] for c in best_circles)
        min_x = min(c[0] - c[2] for c in best_circles)
        max_y = max(c[1] + c[2] for c in best_circles)
        min_y = min(c[1] - c[2] for c in best_circles)
        
        # Try polishing with the original aspect ratios
        for w, h in [(1.0, 1.0), (0.85, 1.15), (0.75, 1.25)]:
            if time.time() > deadline - 2.0:
                break
            polished = _polish(best_circles, w, h, deadline)
            pol_score = _total_radius(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_circles = polished
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
