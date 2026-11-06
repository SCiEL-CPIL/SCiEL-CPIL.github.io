import numpy as np
import scipy.optimize
import time
from scipy.spatial.distance import cdist

N = 50
import itertools

def _score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) != N:
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(np.abs(centers) > 0.5):
        return -np.inf
        
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Vectorized pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _score_fast(centers):
    """Fast score computation using vectorized operations."""
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def generate_hex_config(n_rows, n_cols, dx, dy, offset_x, offset_y, phase=0):
    """Generate a hexagonal lattice configuration with given parameters."""
    centers = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = offset_x + j * dx + (i % 2) * dx / 2.0
            y = offset_y + i * dy
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Keep only the first N points
    if len(centers) > N:
        centers = centers[:N]
    
    # If fewer than N, pad with random
    if len(centers) < N:
        extra = np.random.uniform(-0.4, 0.4, (N - len(centers), 2))
        centers = np.vstack([centers, extra])
    
    return centers

def generate_alternating_hex(row_lengths, dx, dy, offset_x, offset_y):
    """Generate hex packing with alternating row lengths."""
    centers = []
    for i, n in enumerate(row_lengths):
        y = offset_y + i * dy
        # For hex packing, even rows start at offset_x, odd rows at offset_x + dx/2
        x_start = offset_x + (i % 2) * dx / 2.0
        for j in range(n):
            x = x_start + j * dx
            centers.append([x, y])
    
    centers = np.array(centers)
    return centers

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin (evaluator kills at 150s)
    
    best_centers = None
    best_score = -np.inf
    
    # Objective function for optimization
    def objective(x):
        c = x.reshape(-1, 2)
        dist_bound = np.min(0.5 - np.abs(c), axis=1)
        min_bound = np.min(dist_bound)
        
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_pair / 2.0, min_bound)
        return -r
    
    # Strategy 1: Try hexagonal lattice with alternating row lengths
    # The directive suggests rows like 7,6,7,6,7,6,7,4 = 50 circles
    # Let's try various patterns that sum to 50
    
    row_patterns = []
    # Generate patterns with rows of length 6 or 7 (and possibly 5, 8)
    # For 8 rows: combinations of 6s and 7s that sum to 50
    for n_rows in range(7, 11):
        for combo in itertools.product([5, 6, 7, 8], repeat=n_rows):
            if sum(combo) == N:
                row_patterns.append(list(combo))
    
    # Also add specific patterns from the directive
    row_patterns.append([7, 6, 7, 6, 7, 6, 7, 4])
    row_patterns.append([6, 7, 6, 7, 6, 7, 6, 7, 4])
    row_patterns.append([7, 7, 6, 6, 7, 7, 6, 4])
    row_patterns.append([6, 6, 7, 7, 6, 6, 7, 5])
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in row_patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    row_patterns = unique_patterns
    
    for row_lengths in row_patterns:
        if time.time() - start_time > deadline - 5:
            break
        
        n_rows = len(row_lengths)
        max_len = max(row_lengths)
        min_len = min(row_lengths)
        
        # For hex packing: dy = dx * sqrt(3)/2
        # Total height: (n_rows-1) * dy
        # We want total height <= 1.0 (from -0.5 to 0.5)
        # So (n_rows-1) * dx * sqrt(3)/2 <= 1.0
        # dx <= 2.0 / ((n_rows-1) * sqrt(3))
        
        # Try a range of dx values
        dx_max = 2.0 / ((n_rows - 1) * np.sqrt(3.0))
        
        for dx_frac in [0.85, 0.90, 0.95, 1.0, 1.05]:
            dx = dx_max * dx_frac
            dy = dx * np.sqrt(3.0) / 2.0
            
            # Center the pattern vertically
            total_h = (n_rows - 1) * dy
            offset_y = -total_h / 2.0
            
            # For horizontal centering, we need to consider the max width
            # Each row i has row_lengths[i] circles
            # Even rows (i%2==0) start at offset_x
            # Odd rows (i%2==1) start at offset_x + dx/2
            # Width of row i: (row_lengths[i] - 1) * dx + (dx/2 if i%2==1 else 0)
            
            # To center, we want the overall bounding box centered at 0
            # Let's compute the leftmost and rightmost x for each row
            left_edges = []
            right_edges = []
            for i, n in enumerate(row_lengths):
                x_start = (i % 2) * dx / 2.0
                left_edges.append(x_start)
                right_edges.append(x_start + (n - 1) * dx)
            
            min_x = min(left_edges)
            max_x = max(right_edges)
            center_x = (min_x + max_x) / 2.0
            offset_x = -center_x
            
            centers = generate_alternating_hex(row_lengths, dx, dy, offset_x, offset_y)
            
            # Clip to square
            centers = np.clip(centers, -0.5, 0.5)
            
            s = _score_fast(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
    
    # Also try standard hex grid as before
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95
    row_height = d * np.sqrt(3.0) / 2.0
    n_rows = int(np.ceil(1.0 / row_height)) + 2
    n_cols = int(np.ceil(1.0 / d)) + 2
    
    centers_list = []
    for i in range(n_rows):
        y = -0.5 + i * row_height
        if i % 2 == 0:
            x_start = -0.5
        else:
            x_start = -0.5 + d / 2.0
        for j in range(n_cols):
            x = x_start + j * d
            centers_list.append([x, y])
    
    centers_arr = np.array(centers_list)
    centers_arr = np.clip(centers_arr, -0.5, 0.5)
    
    if len(centers_arr) > N:
        dist_to_boundary = np.min(0.5 - np.abs(centers_arr), axis=1)
        sort_idx = np.argsort(dist_to_boundary)
        centers_arr = centers_arr[sort_idx[-N:]]
    elif len(centers_arr) < N:
        extra = np.random.uniform(-0.4, 0.4, (N - len(centers_arr), 2))
        centers_arr = np.vstack([centers_arr, extra])
    
    s = _score_fast(centers_arr)
    if s > best_score:
        best_score = s
        best_centers = centers_arr.copy()
    
    # If no valid configuration found, fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Strategy 2: Local optimization using Powell
    if time.time() - start_time < deadline - 10:
        try:
            res = scipy.optimize.minimize(
                objective,
                best_centers.flatten(),
                method='Powell',
                options={'maxiter': 300, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            opt_centers = res.x.reshape(-1, 2)
            opt_score = -res.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # Strategy 3: Nelder-Mead polish
    if time.time() - start_time < deadline - 5:
        try:
            res2 = scipy.optimize.minimize(
                objective,
                best_centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if -res2.fun > best_score:
                best_score = -res2.fun
                best_centers = res2.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Strategy 4: Random perturbation local search
    if time.time() - start_time < deadline - 3:
        centers = best_centers.copy()
        step_size = 0.01
        patience = 0
        max_patience = 30
        local_best = best_score
        
        for it in range(2000):
            if time.time() - start_time > deadline - 1:
                break
            
            if patience > max_patience:
                centers = best_centers.copy()
                step_size *= 1.3
                patience = 0
            
            noise = np.random.randn(N, 2) * step_size
            new_centers = centers + noise
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _score_fast(new_centers)
            
            if new_score > local_best:
                local_best = new_score
                best_score = new_score
                best_centers = new_centers.copy()
                centers = new_centers.copy()
                patience = 0
                step_size *= 0.95
            else:
                patience += 1
                step_size *= 0.92
    
    # Final clip and return
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
