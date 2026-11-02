"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(circles):
    """Compute the sum of radii for a set of circles, return 0 if infeasible."""
    if circles.shape[0] != N:
        return 0.0
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii are positive
    if np.any(rs <= 0):
        return 0.0
    
    # Find bounding box
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0:
        return 0.0
    
    # Perimeter constraint
    perimeter = 2 * (width + height)
    if perimeter > 4.0 + 1e-9:
        return 0.0
    
    # Check pairwise disjointness
    n = len(rs)
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            r_sum = rs[i] + rs[j]
            if dist_sq < (r_sum - 1e-9)**2:
                return 0.0
    
    return np.sum(rs)

def _make_feasible(circles, max_iter=100):
    """Try to make a circle configuration feasible by shrinking radii and adjusting positions."""
    circles = circles.copy()
    for _ in range(max_iter):
        # Check feasibility
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Check pairwise
        feasible = True
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                r_sum = rs[i] + rs[j]
                if dist_sq < (r_sum - 1e-12)**2:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Check perimeter
            x_min = np.min(xs - rs)
            x_max = np.max(xs + rs)
            y_min = np.min(ys - rs)
            y_max = np.max(ys + rs)
            width = x_max - x_min
            height = y_max - y_min
            perimeter = 2 * (width + height)
            if perimeter <= 4.0 + 1e-9:
                return circles
        
        # Shrink all radii slightly
        circles[:, 2] *= 0.98
    
    return circles

def _optimize_single_aspect(w, h, initial_circles=None):
    """Optimize circle positions and radii for fixed aspect ratio w x h (perimeter 4)."""
    # w + h = 2, so h = 2 - w
    h = 2.0 - w
    
    if initial_circles is None:
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.5 * min(w, h) / side
                cells.append([cx, cy, r])
        initial_circles = np.array(cells[:N])
    
    # Optimize using SLSQP
    n = N
    x0 = initial_circles.flatten()
    
    # Bounds: x in [0, w], y in [0, h], r in [0.01, min(w,h)/2]
    r_max = min(w, h) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.01, r_max))
    
    # Constraints
    constraints = []
    
    # Perimeter constraint is handled by the fixed w, h
    # But we need to ensure circles stay within the rectangle
    # x - r >= 0, x + r <= w, y - r >= 0, y + r <= h
    for i in range(n):
        idx = i * 3
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx] - x[idx+2]})
        # w - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=idx, w=w: w - x[idx] - x[idx+2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx+1] - x[idx+2]})
        # h - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=idx, h=h: h - x[idx+1] - x[idx+2]})
    
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i+1, n):
            idx_i = i * 3
            idx_j = j * 3
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx_i=idx_i, idx_j=idx_j: 
                    (x[idx_i] - x[idx_j])**2 + (x[idx_i+1] - x[idx_j+1])**2 - (x[idx_i+2] + x[idx_j+2])**2
            })
    
    def objective(x):
        circles = x.reshape(n, 3)
        return -np.sum(circles[:, 2])
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 200, 'ftol': 1e-12})
    
    if result.success or _compute_score(result.x.reshape(n, 3)) > 0:
        return result.x.reshape(n, 3)
    
    return initial_circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_circles = None
    best_score = 0.0
    
    # Try different aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        
        circles = _optimize_single_aspect(w)
        score = _compute_score(circles)
        
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    # If we haven't found a good solution, use grid fallback
    if best_circles is None or best_score == 0.0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
