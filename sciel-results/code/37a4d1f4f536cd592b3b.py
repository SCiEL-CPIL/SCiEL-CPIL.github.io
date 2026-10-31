"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _perimeter(w, h):
    return 2.0 * (w + h)

def _is_valid(circles, tol=1e-9):
    """Check if circles are disjoint and within bounding box of perimeter <= 4."""
    if len(circles) == 0:
        return True
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w < -tol or h < -tol:
        return False
    
    if _perimeter(w, h) > 4.0 + tol:
        return False
    
    # Check pairwise disjoint
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _objective(x, N_circles):
    """
    x = [w, h, x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    Minimize negative sum of radii (since we maximize sum of radii)
    subject to constraints.
    """
    w = x[0]
    h = x[1]
    
    # Perimeter constraint: 2(w+h) <= 4 => w+h <= 2
    if w <= 0 or h <= 0:
        return 1e10
    
    sum_r = 0.0
    for i in range(N_circles):
        offset = 2 + i * 3
        xi = x[offset]
        yi = x[offset + 1]
        ri = x[offset + 2]
        
        if ri <= 0:
            return 1e10
        
        # Circle must be within [0, w] x [0, h]
        if xi - ri < -1e-6 or xi + ri > w + 1e-6:
            return 1e10
        if yi - ri < -1e-6 or yi + ri > h + 1e-6:
            return 1e10
        
        sum_r += ri
    
    # Pairwise disjoint
    for i in range(N_circles):
        off_i = 2 + i * 3
        xi = x[off_i]
        yi = x[off_i + 1]
        ri = x[off_i + 2]
        for j in range(i + 1, N_circles):
            off_j = 2 + j * 3
            xj = x[off_j]
            yj = x[off_j + 1]
            rj = x[off_j + 2]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < ri + rj - 1e-6:
                return 1e10
    
    return -sum_r

def _make_initial_guess(w, h, N_circles):
    """Create initial guess for optimization."""
    x = np.zeros(2 + 3 * N_circles)
    x[0] = w
    x[1] = h
    
    # Arrange in a grid
    cols = int(np.ceil(np.sqrt(N_circles * w / h))) if h > 0 else N_circles
    rows = int(np.ceil(N_circles / cols))
    
    cell_w = w / cols
    cell_h = h / rows
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= N_circles:
                break
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            cr = min(cell_w, cell_h) / 2 * 0.9
            off = 2 + idx * 3
            x[off] = cx
            x[off + 1] = cy
            x[off + 2] = cr
            idx += 1
    
    return x

def _optimize_from(w, h, N_circles, max_iter=200):
    """Run optimization from initial guess."""
    x0 = _make_initial_guess(w, h, N_circles)
    
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for i in range(N_circles):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.001, min(w, h) / 2))
    
    result = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.success and result.fun < 1e9:
        x_opt = result.x
        w_opt = x_opt[0]
        h_opt = x_opt[1]
        circles = []
        for i in range(N_circles):
            off = 2 + i * 3
            circles.append([x_opt[off], x_opt[off + 1], x_opt[off + 2]])
        circles = np.array(circles)
        
        # Normalize: shift so min corner is at origin, scale so perimeter is exactly 4
        min_x = np.min(circles[:, 0] - circles[:, 2])
        max_x = np.max(circles[:, 0] + circles[:, 2])
        min_y = np.min(circles[:, 1] - circles[:, 2])
        max_y = np.max(circles[:, 1] + circles[:, 2])
        
        bw = max_x - min_x
        bh = max_y - min_y
        perim = 2 * (bw + bh)
        
        if perim > 0 and perim <= 4.0:
            # Scale to make perimeter exactly 4
            scale = 4.0 / perim
            circles[:, 0] = (circles[:, 0] - min_x) * scale
            circles[:, 1] = (circles[:, 1] - min_y) * scale
            circles[:, 2] = circles[:, 2] * scale
            
            if _is_valid(circles):
                return circles, np.sum(circles[:, 2])
    
    return None, 0.0

def _grid_search():
    """Try various aspect ratios."""
    best_circles = None
    best_sum = 0.0
    
    # Try a range of aspect ratios w/h
    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        # w + h = 2 (for perimeter 4)
        # h = w / ratio => w + w/ratio = 2 => w(1 + 1/ratio) = 2
        w = 2.0 * ratio / (1.0 + ratio)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        circles, s = _optimize_from(w, h, N)
        if circles is not None and s > best_sum:
            best_sum = s
            best_circles = circles
    
    return best_circles, best_sum

def _refine(circles, max_iter=100):
    """Refine a valid packing."""
    # Extract w, h from bounding box
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    
    w = max_x - min_x
    h = max_y - min_y
    
    # Shift to origin
    c_shifted = circles.copy()
    c_shifted[:, 0] -= min_x
    c_shifted[:, 1] -= min_y
    
    # Build initial x for refinement
    x0 = np.zeros(2 + 3 * N)
    x0[0] = w
    x0[1] = h
    for i in range(N):
        off = 2 + i * 3
        x0[off] = c_shifted[i, 0]
        x0[off + 1] = c_shifted[i, 1]
        x0[off + 2] = c_shifted[i, 2]
    
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for i in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.001, min(w, h) / 2))
    
    result = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    if result.success and result.fun < 1e9:
        x_opt = result.x
        w_opt = x_opt[0]
        h_opt = x_opt[1]
        circles_opt = []
        for i in range(N):
            off = 2 + i * 3
            circles_opt.append([x_opt[off], x_opt[off + 1], x_opt[off + 2]])
        circles_opt = np.array(circles_opt)
        
        # Normalize
        min_x = np.min(circles_opt[:, 0] - circles_opt[:, 2])
        max_x = np.max(circles_opt[:, 0] + circles_opt[:, 2])
        min_y = np.min(circles_opt[:, 1] - circles_opt[:, 2])
        max_y = np.max(circles_opt[:, 1] + circles_opt[:, 2])
        
        bw = max_x - min_x
        bh = max_y - min_y
        perim = 2 * (bw + bh)
        
        if perim > 0 and perim <= 4.0:
            scale = 4.0 / perim
            circles_opt[:, 0] = (circles_opt[:, 0] - min_x) * scale
            circles_opt[:, 1] = (circles_opt[:, 1] - min_y) * scale
            circles_opt[:, 2] = circles_opt[:, 2] * scale
            
            if _is_valid(circles_opt):
                return circles_opt, np.sum(circles_opt[:, 2])
    
    return circles, np.sum(circles[:, 2])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_circles = None
    best_sum = 0.0
    
    # First, try grid search over aspect ratios
    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        if time.time() > deadline:
            break
        
        w = 2.0 * ratio / (1.0 + ratio)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        circles, s = _optimize_from(w, h, N, max_iter=100)
        if circles is not None and s > best_sum:
            best_sum = s
            best_circles = circles
    
    # Refine the best solution
    if best_circles is not None and time.time() < deadline:
        refined, s = _refine(best_circles, max_iter=150)
        if s > best_sum:
            best_sum = s
            best_circles = refined
    
    # Try more aspect ratios if time permits
    if time.time() < deadline:
        for ratio in [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.6, 1.8]:
            if time.time() > deadline:
                break
            
            w = 2.0 * ratio / (1.0 + ratio)
            h = 2.0 - w
            
            if w <= 0 or h <= 0:
                continue
            
            circles, s = _optimize_from(w, h, N, max_iter=80)
            if circles is not None and s > best_sum:
                best_sum = s
                best_circles = circles
        
        # Final refinement
        if best_circles is not None and time.time() < deadline - 10:
            refined, s = _refine(best_circles, max_iter=100)
            if s > best_sum:
                best_sum = s
                best_circles = refined
    
    # If we still don't have a valid solution, use a simple grid
    if best_circles is None or not _is_valid(best_circles):
        # Fallback: simple grid in a square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        # Normalize to perimeter 4
        min_x = np.min(best_circles[:, 0] - best_circles[:, 2])
        max_x = np.max(best_circles[:, 0] + best_circles[:, 2])
        min_y = np.min(best_circles[:, 1] - best_circles[:, 2])
        max_y = np.max(best_circles[:, 1] + best_circles[:, 2])
        bw = max_x - min_x
        bh = max_y - min_y
        perim = 2 * (bw + bh)
        if perim > 0:
            scale = 4.0 / perim
            best_circles[:, 0] = (best_circles[:, 0] - min_x) * scale
            best_circles[:, 1] = (best_circles[:, 1] - min_y) * scale
            best_circles[:, 2] = best_circles[:, 2] * scale
    
    return best_circles
# EVOLVE-BLOCK-END
