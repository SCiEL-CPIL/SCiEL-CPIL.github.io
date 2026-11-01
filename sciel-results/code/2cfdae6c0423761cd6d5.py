"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 110  # safety margin before 120s limit


def _check_validity(circles, w, h):
    """Check if circles are valid: within [0,w]x[0,h], disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -1e-9 or x + r > w + 1e-9:
            return False
        if y - r < -1e-9 or y + r > h + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < min_dist ** 2 - 1e-9:
                return False
    return True


def _make_initial_grid(n_cols, n_rows, w, h):
    """Create a grid-based initial configuration scaled to (w,h)."""
    cells = []
    for i in range(n_rows):
        for j in range(n_cols):
            if len(cells) >= N:
                break
            x = (j + 0.5) / n_cols * w
            y = (i + 0.5) / n_rows * h
            r = 0.5 / max(n_cols, n_rows) * min(w, h) * 0.95
            cells.append((x, y, r))
    return cells


def _optimize_positions(circles_init, w_init, h_init, time_budget=20.0):
    """
    Stage 1: Optimize positions and aspect ratio with fixed relative radii.
    Parameters: [x1,y1,...,xn,yn, w] where h = 2-w, radii scaled proportionally.
    """
    n = N
    r_init = np.array([c[2] for c in circles_init])
    r_scale = np.sum(r_init)
    
    def obj(params):
        pos = params[:2 * n]
        w_val = params[2 * n]
        h_val = 2.0 - w_val
        
        if w_val <= 0.05 or h_val <= 0.05:
            return 1e10
        
        # Scale radii to maintain relative proportions, fit in box
        scale_factor = min(w_val / 2.0, h_val / 2.0)
        radii = r_init * scale_factor
        
        penalty = 0.0
        for i in range(n):
            x = pos[2 * i]
            y = pos[2 * i + 1]
            r = radii[i]
            if x < r:
                penalty += 1e4 * (r - x) ** 2
            if x > w_val - r:
                penalty += 1e4 * (x - (w_val - r)) ** 2
            if y < r:
                penalty += 1e4 * (r - y) ** 2
            if y > h_val - r:
                penalty += 1e4 * (y - (h_val - r)) ** 2
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((pos[2*i] - pos[2*j]) ** 2 + (pos[2*i+1] - pos[2*j+1]) ** 2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += 1e4 * (min_dist - dist) ** 2
        
        return penalty
    
    # Initial positions
    x0 = np.zeros(2 * n + 1)
    for i in range(n):
        x0[2 * i] = circles_init[i][0]
        x0[2 * i + 1] = circles_init[i][1]
    x0[2 * n] = w_init
    
    bounds = []
    for i in range(n):
        bounds.append((0, 2))
        bounds.append((0, 2))
    bounds.append((0.1, 1.9))
    
    start_time = time.time()
    
    try:
        result = minimize(
            obj, x0, method='SLSQP', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.fun < 1e-6:
            pos = result.x[:2 * n]
            w_val = result.x[2 * n]
            h_val = 2.0 - w_val
            scale_factor = min(w_val / 2.0, h_val / 2.0)
            radii = r_init * scale_factor
            
            circles = np.column_stack([pos[0::2], pos[1::2], radii])
            if _check_validity(circles, w_val, h_val):
                return circles, w_val, h_val
        return None
    except Exception:
        return None


def _optimize_radii(circles_init, w, h, time_budget=20.0):
    """
    Stage 2: Optimize radii given fixed positions and container.
    This is a convex problem: maximize sum(r) subject to r_i + r_j <= dist_ij
    and r_i <= min(x_i, w-x_i, y_i, h-y_i) for each circle.
    """
    n = N
    centers = np.array([(c[0], c[1]) for c in circles_init])
    
    # Compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
            dists[i, j] = d
            dists[j, i] = d
    
    # Max radius for each circle based on boundary
    max_r = np.zeros(n)
    for i in range(n):
        max_r[i] = min(centers[i][0], w - centers[i][0], centers[i][1], h - centers[i][1])
        max_r[i] = max(max_r[i], 0.0)
    
    def obj(radii):
        # Minimize negative sum of radii + penalty for violations
        penalty = 0.0
        for i in range(n):
            if radii[i] > max_r[i]:
                penalty += 1e4 * (radii[i] - max_r[i]) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                if radii[i] + radii[j] > dists[i, j]:
                    penalty += 1e4 * (radii[i] + radii[j] - dists[i, j]) ** 2
        return penalty - np.sum(radii)
    
    r0 = np.array([c[2] * 0.9 for c in circles_init])
    bounds = [(0, max_r[i]) for i in range(n)]
    
    start_time = time.time()
    
    try:
        result = minimize(
            obj, r0, method='SLSQP', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        
        radii = result.x
        # Clip to valid range
        for i in range(n):
            radii[i] = min(radii[i], max_r[i])
        
        circles = np.column_stack([centers[:, 0], centers[:, 1], radii])
        if _check_validity(circles, w, h):
            return circles
        return None
    except Exception:
        return None


def _full_optimize(circles_init, w_init, h_init, time_budget=30.0):
    """Full two-stage optimization: positions first, then radii."""
    start = time.time()
    
    # Stage 1: Optimize positions
    result1 = _optimize_positions(circles_init, w_init, h_init, time_budget * 0.5)
    if result1 is None:
        return None
    
    circles_pos, w_val, h_val = result1
    
    # Stage 2: Optimize radii
    if time.time() - start > time_budget * 0.7:
        return circles_pos, w_val, h_val
    
    result2 = _optimize_radii(circles_pos, w_val, h_val, time_budget * 0.3)
    if result2 is not None:
        return result2, w_val, h_val
    
    return circles_pos, w_val, h_val


def construct():
    start_time = time.time()
    best_value = 0.0
    best_circles = None
    best_w = 1.0
    best_h = 1.0
    
    # Try multiple grid configurations and aspect ratios
    configs = [
        # (n_cols, n_rows, w_init, h_init)
        (5, 5, 1.0, 1.0),
        (4, 6, 0.8, 1.2),
        (6, 4, 1.2, 0.8),
        (7, 3, 1.4, 0.6),
        (3, 7, 0.6, 1.4),
        (5, 4, 1.1, 0.9),
        (4, 5, 0.9, 1.1),
    ]
    
    for n_cols, n_rows, w_init, h_init in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Create initial grid
        circles_init = _make_initial_grid(n_cols, n_rows, w_init, h_init)
        
        # Run full optimization
        result = _full_optimize(circles_init, w_init, h_init, time_budget=12.0)
        if result is not None:
            circles, w_val, h_val = result
            val = np.sum(circles[:, 2])
            if val > best_value:
                best_value = val
                best_circles = circles.copy()
                best_w = w_val
                best_h = h_val
    
    # Final refinement: try to squeeze more from the best solution
    if best_circles is not None and time.time() - start_time < DEADLINE - 10:
        refined = _optimize_radii(best_circles, best_w, best_h, time_budget=8.0)
        if refined is not None:
            val = np.sum(refined[:, 2])
            if val > best_value:
                best_circles = refined
                best_value = val
    
    # Verify and return
    if best_circles is not None:
        if _check_validity(best_circles, best_w, best_h):
            perim = 2.0 * (best_w + best_h)
            if perim > 4.0 + 1e-6:
                scale = 4.0 / perim
                best_circles = best_circles * scale
            return best_circles
    
    # Fallback
    circles_init = _make_initial_grid(5, 5, 1.0, 1.0)
    return np.array(circles_init[:N])
