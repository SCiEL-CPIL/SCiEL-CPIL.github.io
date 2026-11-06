"""Improved packing: hexagonal lattice with optimized spacing + local refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _soft_min(values, k=100.0):
    """Approximate min using log-sum-exp."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    min_val = np.min(values)
    # Avoid overflow in exp
    shifted = values - min_val
    log_sum_exp = np.log(np.sum(np.exp(k * shifted))) / k
    return min_val - log_sum_exp

def _soft_max(values, k=100.0):
    """Approximate max using log-sum-exp."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    shifted = values - max_val
    log_sum_exp = np.log(np.sum(np.exp(k * shifted))) / k
    return max_val + log_sum_exp

def _exact_score(centers):
    """Compute the minimum radius (score) for a set of centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: min distance to boundary
    dist_bound = np.min(np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ], axis=1), axis=1)
    min_boundary = np.min(dist_bound)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists) / 2.0
    
    return min(min_boundary, min_dist)

def _smooth_score(centers, k=50.0):
    """Compute a smooth approximation of the score."""
    n = len(centers)
    if n == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return -1e10

    # Boundary constraints
    dist_bound = np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ], axis=1)
    # We want the minimum of the minimums.
    # min_i min_j dist_bound[i,j]
    # This is equivalent to min over all boundary distances.
    all_bound_dists = dist_bound.flatten()
    min_boundary_smooth = _soft_min(all_bound_dists, k)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_smooth = _soft_min(dists, k) / 2.0

    # The score is min(boundary, dist).
    # We approximate min(a, b) with soft_min([a, b])
    return _soft_min(np.array([min_boundary_smooth, min_dist_smooth]), k)

def _build_from_rows(rows, params):
    """Build centers from row structure with given parameters.
    params: [spacing, y_offset, shift_ratio, vert_ratio]
    """
    spacing = params[0]
    offset_y = params[1]
    shift_ratio = params[2] # fraction of spacing to shift odd rows
    vert_ratio = params[3] # fraction of spacing for vertical step (hex is sqrt(3)/2)
    
    n_rows = len(rows)
    total_height = (n_rows - 1) * spacing * vert_ratio
    # Center vertically
    y_base = -total_height / 2 + offset_y
    
    centers = []
    for i, count in enumerate(rows):
        if count > 1:
            # Spread circles across available width
            # We define the row to be centered at 0
            # Positions are relative to the row center
            max_width = (count - 1) * spacing
            x_start = -max_width / 2
            x_positions = x_start + np.arange(count) * spacing
        else:
            x_positions = np.array([0.0])
        
        # Shift odd rows
        if i % 2 == 1:
            x_positions += spacing * shift_ratio
        
        y_pos = y_base + i * spacing * vert_ratio
        for x in x_positions:
            centers.append([x, y_pos])
    return np.array(centers)

def _local_refinement(centers, deadline=None):
    """Local refinement using smooth then exact objective."""
    if deadline is not None and time.time() > deadline:
        return centers
    
    n = len(centers)
    x0 = centers.flatten()
    
    # Pass 1: Smooth objective
    def smooth_obj(x):
        c = x.reshape(-1, 2)
        if np.any(np.abs(c) > 0.5 + 1e-6):
            # Penalize going out of bounds heavily
            return 1e6 + np.sum(np.abs(c) - 0.5)
        return -_smooth_score(c, k=50.0)
        
    try:
        res1 = minimize(smooth_obj, x0, method='Nelder-Mead', 
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        x1 = res1.x
    except Exception:
        x1 = x0
        
    if deadline is not None and time.time() > deadline:
        return x1.reshape(-1, 2)
        
    # Pass 2: Exact objective
    def exact_obj(x):
        c = x.reshape(-1, 2)
        if np.any(np.abs(c) > 0.5 + 1e-6):
            return 1e6 + np.sum(np.abs(c) - 0.5)
        return -_exact_score(c)
        
    try:
        res2 = minimize(exact_obj, x1, method='Nelder-Mead', 
                        options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
        x2 = res2.x
    except Exception:
        x2 = x1
        
    return x2.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try several row configurations
    row_configs = [
        [7, 6, 7, 6, 7, 6, 7, 4],      # 50
        [8, 7, 7, 7, 7, 7, 7],           # 50
        [6, 7, 7, 7, 7, 7, 7, 2],        # 50
        [5, 6, 7, 7, 7, 7, 7, 4],        # 50
        [4, 7, 7, 7, 7, 7, 7, 4],        # 50
        [7, 7, 7, 7, 7, 7, 8],           # 50
        [6, 6, 7, 7, 7, 7, 7, 3],        # 50
        [5, 7, 7, 7, 7, 7, 7, 3],        # 50
        [8, 6, 7, 7, 7, 7, 8],           # 50
        [7, 6, 7, 7, 7, 6, 7, 3],        # 50
        [6, 7, 6, 7, 7, 7, 6, 4],        # 50
        [7, 7, 6, 7, 7, 6, 7, 3],        # 50
        [7, 7, 7, 6, 7, 7, 7, 2],        # 50
        [6, 7, 7, 7, 6, 7, 7, 3],        # 50
    ]
    
    best_centers = None
    best_score = 0.0
    
    for rows in row_configs:
        if sum(rows) != N:
            continue
        if time.time() > deadline:
            break
        
        n_rows = len(rows)
        max_count = max(rows)
        
        # Initial spacing estimate
        s_w = 1.0 / max_count
        s_h = 1.0 / (n_rows * np.sqrt(3) / 2)
        s_init = min(s_w, s_h) * 0.9
        
        # Parameters: spacing, y_offset, shift_ratio, vert_ratio
        bounds = [
            (s_init * 0.8, s_init * 1.2), # spacing
            (-0.1, 0.1),                  # y_offset
            (0.3, 0.7),                   # shift_ratio (0.5 is hex)
            (0.7, 1.0)                    # vert_ratio (sqrt(3)/2 ~ 0.866 is hex)
        ]
        
        def de_objective(params):
            centers = _build_from_rows(rows, params)
            # Check validity
            if np.any(np.abs(centers) > 0.5 + 1e-9):
                return 1e10
            # Use smooth score for DE to navigate better
            return -_smooth_score(centers, k=20.0)
        
        try:
            de_result = differential_evolution(
                de_objective, bounds, 
                maxiter=80, popsize=15, tol=1e-6,
                seed=42, polish=False, init='latinhypercube'
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        best_params = de_result.x
        centers = _build_from_rows(rows, best_params)
        
        # Local refinement
        try:
            centers = _local_refinement(centers, deadline=deadline)
        except Exception:
            pass
        
        # Check validity and score
        if np.any(~np.isfinite(centers)):
            continue
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            continue
        
        s = _exact_score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
    
    # If no valid result, fall back to grid
    if best_centers is None or best_score <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Final check
    if time.time() > deadline:
        return best_centers
    
    # One more refinement pass if time allows
    try:
        refined = _local_refinement(best_centers, deadline=deadline)
        if np.all(np.isfinite(refined)) and np.all(np.abs(refined) <= 0.5 + 1e-9):
            s2 = _exact_score(refined)
            if s2 > best_score:
                best_centers = refined
    except Exception:
        pass
    
    return best_centers
# EVOLVE-BLOCK-END
