"""Circle packing in perimeter-4 rectangle.
Maximize sum of 21 disjoint circle radii inside a rectangle with perimeter <= 4.

Strategy:
1. Try a variety of parametric layouts:
   - Regular grids (nc x nr)
   - Hexagonal (staggered) grids
   - Mixed-row layouts (different counts per row)
2. For each layout, compute the optimal rectangle aspect ratio (w+h=2) that
   maximizes the minimum radius.
3. Use a local optimization step (gradient-free, e.g., Nelder-Mead or simple
   coordinate descent) to fine-tune circle positions for the best few layouts.
4. Return the layout with the maximum sum of radii.

Key insight: For a given set of relative positions (normalized), the max radius
is determined by the minimum distance to the boundary and between circles.
We can optimize the aspect ratio analytically for regular patterns, and use
numerical optimization for irregular ones.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _hexagonal_layout(n_cols, n_rows):
    """Generate centers for a hexagonal (staggered) grid.
    Returns (xs, ys) as arrays for unit radius (r=1) layout.
    The bounding box is [0, W] x [0, H] where W, H are computed.
    """
    xs = []
    ys = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * 2.0 + (1.0 if row % 2 == 1 else 0.0)
            y = row * np.sqrt(3.0)
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    # Shift so min is at 1 (for unit radius)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _grid_layout(n_cols, n_rows):
    xs = []
    ys = []
    for row in range(n_rows):
        for col in range(n_cols):
            xs.append(col * 2.0 + 1.0)
            ys.append(row * 2.0 + 1.0)
    xs = np.array(xs)
    ys = np.array(ys)
    W = (n_cols - 1) * 2.0 + 2.0
    H = (n_rows - 1) * 2.0 + 2.0
    return xs, ys, W, H


def _mixed_row_layout(row_counts):
    """Generate a staggered layout with varying row counts.
    row_counts: list of integers, number of circles per row.
    Returns (xs, ys, W, H) for unit radius.
    """
    xs = []
    ys = []
    for row_idx, n_circ in enumerate(row_counts):
        y = row_idx * np.sqrt(3.0)
        offset = 1.0 if row_idx % 2 == 1 else 0.0
        for col in range(n_circ):
            x = col * 2.0 + offset
            xs.append(x)
            ys.append(y)
    if len(xs) == 0:
        return None
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _compute_optimal_layout(xs, ys, W, H, n_circles):
    """Given a layout with n_circles positions, compute the optimal rectangle
    and radius. Returns (sum_radii, actual_positions, w, h, r)."""
    if len(xs) < n_circles:
        return 0.0, None, 0, 0, 0
    
    # Use only the first n_circles
    xs_sub = xs[:n_circles]
    ys_sub = ys[:n_circles]
    
    # Recompute bounding box for the subset
    W_sub = xs_sub.max() - xs_sub.min() + 2.0
    H_sub = ys_sub.max() - ys_sub.min() + 2.0
    
    # Optimal aspect ratio: w/W = h/H, w+h=2
    # w = 2W/(W+H), h = 2H/(W+H)
    # r = 2/(W+H)
    r = 2.0 / (W_sub + H_sub)
    w = 2.0 * W_sub / (W_sub + H_sub)
    h = 2.0 * H_sub / (W_sub + H_sub)
    
    # Scale positions
    x_scale = r
    actual_x = (xs_sub - xs_sub.min()) * x_scale + r
    actual_y = (ys_sub - ys_sub.min()) * x_scale + r
    
    # Verify constraints
    perim = 2.0 * (w + h)
    if perim > 4.0 + 1e-9:
        return 0.0, None, 0, 0, 0
    
    # Verify circles are inside and disjoint
    for i in range(n_circles):
        if actual_x[i] < r - 1e-9 or actual_x[i] > w - r + 1e-9:
            return 0.0, None, 0, 0, 0
        if actual_y[i] < r - 1e-9 or actual_y[i] > h - r + 1e-9:
            return 0.0, None, 0, 0, 0
    
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt((actual_x[i] - actual_x[j])**2 + (actual_y[i] - actual_y[j])**2)
            if dist < 2.0 * r - 1e-9:
                return 0.0, None, 0, 0, 0
    
    sum_radii = n_circles * r
    return sum_radii, np.column_stack([actual_x, actual_y, r]), w, h, r


def _optimize_positions(x0, y0, w, h, r, max_iters=200):
    """Try to improve the sum of radii by adjusting positions and radius.
    Uses a simple local search."""
    n = len(x0)
    best_x = x0.copy()
    best_y = y0.copy()
    best_r = r
    best_sum = n * r
    
    # Objective: maximize r subject to constraints
    # We'll do a simple gradient-free optimization on r and positions
    
    def objective(params):
        # params: [r, x0, y0, x1, y1, ..., xn-1, yn-1]
        r_val = params[0]
        if r_val <= 0:
            return 1e10
        xs = params[1::2]
        ys = params[2::2]
        
        # Boundary constraints
        penalty = 0.0
        for i in range(n):
            if xs[i] < r_val:
                penalty += (r_val - xs[i])**2
            if xs[i] > w - r_val:
                penalty += (xs[i] - (w - r_val))**2
            if ys[i] < r_val:
                penalty += (r_val - ys[i])**2
            if ys[i] > h - r_val:
                penalty += (ys[i] - (h - r_val))**2
        
        # Disjointness constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if dist < 2.0 * r_val:
                    penalty += (2.0 * r_val - dist)**2
        
        # Maximize r, so minimize -r + penalty
        return -r_val + 100.0 * penalty
    
    # Initial params
    params0 = np.concatenate([[r], x0, y0])
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.success or result.fun < -best_r + 1e-6:
            r_opt = result.x[0]
            x_opt = result.x[1::2]
            y_opt = result.x[2::2]
            
            # Verify
            valid = True
            for i in range(n):
                if x_opt[i] < r_opt - 1e-8 or x_opt[i] > w - r_opt + 1e-8:
                    valid = False
                    break
                if y_opt[i] < r_opt - 1e-8 or y_opt[i] > h - r_opt + 1e-8:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2)
                        if dist < 2.0 * r_opt - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid and r_opt > best_r:
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt
                best_sum = n * r_opt
    except Exception:
        pass
    
    return best_sum, best_x, best_y, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_sum = 0.0
    best_result = None
    
    # Try various layouts
    candidates = []
    
    # Grid layouts
    for nc in range(1, 12):
        for nr in range(1, 12):
            if nc * nr >= N:
                candidates.append(('grid', nc, nr))
    
    # Hexagonal layouts
    for nc in range(1, 12):
        for nr in range(1, 12):
            if nc * nr >= N:
                candidates.append(('hex', nc, nr))
    
    # Mixed row layouts (staggered with varying counts)
    # Try rows with counts like [4,4,4,4,3,2], [5,4,4,4,3,1], etc.
    base_counts = [
        [4, 4, 4, 4, 3, 2],
        [5, 4, 4, 4, 3, 1],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [5, 5, 4, 3, 2, 2],
        [6, 4, 4, 3, 2, 2],
        [4, 5, 4, 4, 3, 1],
        [3, 4, 4, 5, 3, 2],
        [5, 4, 5, 3, 2, 2],
        [4, 4, 5, 4, 3, 1],
    ]
    for counts in base_counts:
        if sum(counts) >= N:
            candidates.append(('mixed', counts))
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 10.0:
            break
        
        try:
            if cand[0] == 'grid':
                _, nc, nr = cand
                xs, ys, W, H = _grid_layout(nc, nr)
            elif cand[0] == 'hex':
                _, nc, nr = cand
                xs, ys, W, H = _hexagonal_layout(nc, nr)
            elif cand[0] == 'mixed':
                _, counts = cand
                res = _mixed_row_layout(counts)
                if res is None:
                    continue
                xs, ys, W, H = res
            else:
                continue
            
            s, result, w, h, r = _compute_optimal_layout(xs, ys, W, H, N)
            
            if s > best_sum:
                best_sum = s
                best_result = result
            
            # If this is a promising layout, try local optimization
            if s > best_sum * 0.95 and time.time() < deadline - 20.0:
                # Extract positions from the best layout
                xs_sub = xs[:N]
                ys_sub = ys[:N]
                W_sub = xs_sub.max() - xs_sub.min() + 2.0
                H_sub = ys_sub.max() - ys_sub.min() + 2.0
                r_opt = 2.0 / (W_sub + H_sub)
                w_opt = 2.0 * W_sub / (W_sub + H_sub)
                h_opt = 2.0 * H_sub / (W_sub + H_sub)
                
                x_scale = r_opt
                x0 = (xs_sub - xs_sub.min()) * x_scale + r_opt
                y0 = (ys_sub - ys_sub.min()) * x_scale + r_opt
                
                opt_sum, x_opt, y_opt, r_final = _optimize_positions(
                    x0, y0, w_opt, h_opt, r_opt, max_iters=100
                )
                
                if opt_sum > best_sum:
                    best_sum = opt_sum
                    best_result = np.column_stack([x_opt, y_opt, r_final])
                    
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
