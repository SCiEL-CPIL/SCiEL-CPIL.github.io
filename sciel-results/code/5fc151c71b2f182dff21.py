"""Circle packing in perimeter-4 rectangle.
Maximize sum of 21 disjoint circle radii inside a rectangle with perimeter <= 4.

Strategy:
1. Try a variety of parametric layouts:
   - Regular grids (nc x nr)
   - Hexagonal (staggered) grids
   - Mixed-row layouts (different counts per row)
   - Randomized layouts
2. For each layout, compute the optimal rectangle aspect ratio (w+h=2) that
   maximizes the minimum radius.
3. Use a vectorized local optimization step to fine-tune circle positions.
4. Return the layout with the maximum sum of radii.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _hexagonal_layout(n_cols, n_rows):
    """Generate centers for a hexagonal (staggered) grid.
    Returns (xs, ys) as arrays for unit radius (r=1) layout.
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
    """Generate a staggered layout with varying row counts."""
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


def _random_layout(n, seed=42):
    """Generate a random layout with n circles."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0, 1, n)
    ys = rng.uniform(0, 1, n)
    # Normalize to have unit radius spacing
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _compute_optimal_layout(xs, ys, n_circles):
    """Given a layout, compute optimal rectangle and radius.
    Returns (sum_radii, actual_positions, w, h, r)."""
    if len(xs) < n_circles:
        return 0.0, None, 0, 0, 0
    
    xs_sub = xs[:n_circles]
    ys_sub = ys[:n_circles]
    
    W_sub = xs_sub.max() - xs_sub.min() + 2.0
    H_sub = ys_sub.max() - ys_sub.min() + 2.0
    
    r = 2.0 / (W_sub + H_sub)
    w = 2.0 * W_sub / (W_sub + H_sub)
    h = 2.0 * H_sub / (W_sub + H_sub)
    
    actual_x = (xs_sub - xs_sub.min()) * r + r
    actual_y = (ys_sub - ys_sub.min()) * r + r
    
    # Verify constraints
    if 2.0 * (w + h) > 4.0 + 1e-9:
        return 0.0, None, 0, 0, 0
    
    if np.any(actual_x < r - 1e-9) or np.any(actual_x > w - r + 1e-9):
        return 0.0, None, 0, 0, 0
    if np.any(actual_y < r - 1e-9) or np.any(actual_y > h - r + 1e-9):
        return 0.0, None, 0, 0, 0
    
    # Check disjointness
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt((actual_x[i] - actual_x[j])**2 + (actual_y[i] - actual_y[j])**2)
            if dist < 2.0 * r - 1e-9:
                return 0.0, None, 0, 0, 0
    
    return n_circles * r, np.column_stack([actual_x, actual_y, r]), w, h, r


def _is_valid(x, y, r, w, h, tol=1e-8):
    """Check if circles are valid (inside rectangle and disjoint)."""
    n = len(x)
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < 2.0 * r - tol:
                return False
    return True


def _objective(params, w, h):
    """Objective function for optimization: minimize -r + penalty."""
    n = (len(params) - 1) // 2
    r_val = params[0]
    if r_val <= 0:
        return 1e10
    xs = params[1::2]
    ys = params[2::2]
    
    # Boundary penalty
    penalty = 0.0
    penalty += np.sum(np.maximum(0, r_val - xs)**2)
    penalty += np.sum(np.maximum(0, xs - (w - r_val))**2)
    penalty += np.sum(np.maximum(0, r_val - ys)**2)
    penalty += np.sum(np.maximum(0, ys - (h - r_val))**2)
    
    # Disjointness penalty
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < 2.0 * r_val:
                penalty += (2.0 * r_val - dist)**2
    
    return -r_val + 100.0 * penalty


def _optimize_positions(x0, y0, w, h, r, max_iters=300, tol=1e-8):
    """Local optimization to improve radius."""
    n = len(x0)
    best_x = x0.copy()
    best_y = y0.copy()
    best_r = r
    best_sum = n * r
    
    params0 = np.concatenate([[r], x0, y0])
    
    try:
        result = minimize(_objective, params0, args=(w, h), method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': 1e-7, 'fatol': 1e-7})
        r_opt = result.x[0]
        x_opt = result.x[1::2]
        y_opt = result.x[2::2]
        
        if _is_valid(x_opt, y_opt, r_opt, w, h, tol) and r_opt > best_r:
            best_x = x_opt
            best_y = y_opt
            best_r = r_opt
            best_sum = n * r_opt
    except Exception:
        pass
    
    return best_sum, best_x, best_y, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_sum = 0.0
    best_result = None
    
    # Collect candidate layouts
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
    
    # Mixed row layouts - expanded set
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
        [7, 5, 4, 3, 2],
        [6, 6, 4, 3, 2],
        [5, 5, 5, 4, 2],
        [4, 4, 4, 4, 4, 1],
        [6, 5, 4, 4, 2],
        [5, 5, 4, 4, 3],
        [7, 4, 4, 4, 2],
        [6, 4, 5, 4, 2],
        [5, 6, 4, 4, 2],
        [4, 5, 5, 4, 3],
    ]
    for counts in base_counts:
        if sum(counts) >= N:
            candidates.append(('mixed', counts))
    
    # Random layouts with different seeds
    for seed in range(20):
        candidates.append(('random', seed))
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 15.0:
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
            elif cand[0] == 'random':
                _, seed = cand
                xs, ys, W, H = _random_layout(N, seed)
            else:
                continue
            
            s, result, w, h, r = _compute_optimal_layout(xs, ys, N)
            
            if s > best_sum:
                best_sum = s
                best_result = result
            
            # Try local optimization for promising layouts
            if s > best_sum * 0.98 and time.time() < deadline - 25.0:
                xs_sub = xs[:N]
                ys_sub = ys[:N]
                W_sub = xs_sub.max() - xs_sub.min() + 2.0
                H_sub = ys_sub.max() - ys_sub.min() + 2.0
                r_opt = 2.0 / (W_sub + H_sub)
                w_opt = 2.0 * W_sub / (W_sub + H_sub)
                h_opt = 2.0 * H_sub / (W_sub + H_sub)
                
                x0 = (xs_sub - xs_sub.min()) * r_opt + r_opt
                y0 = (ys_sub - ys_sub.min()) * r_opt + r_opt
                
                opt_sum, x_opt, y_opt, r_final = _optimize_positions(
                    x0, y0, w_opt, h_opt, r_opt, max_iters=200
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
