"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 50

def _score_from_centers(centers):
    """Compute the common radius for a set of centers inside [-0.5, 0.5]^2."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check boundaries
    dist_to_bound = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1), axis=1)
    min_boundary = np.min(dist_to_bound)
    if min_boundary < -1e-12:
        return 0.0
    # Check pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    if min_pair < -1e-12:
        return 0.0
    return min(min_boundary, min_pair)

def _generate_hex_packing(params, row_lengths):
    """
    Generate hexagonal lattice packing.
    params: [sx, sy, x_shift_odd, x_global, y_global]
    row_lengths: list of integers for each row's circle count.
    """
    sx, sy, x_shift_odd, x_global, y_global = params
    n_rows = len(row_lengths)
    
    centers = []
    for i, length in enumerate(row_lengths):
        if length == 0:
            continue
        y = y_global + (i - (n_rows - 1) / 2.0) * sy
        shift = x_shift_odd if i % 2 != 0 else 0.0
        x_start = x_global + shift - sx * (length - 1) / 2.0
        for j in range(length):
            x = x_start + j * sx
            centers.append([x, y])
            
    centers = np.array(centers)
    if len(centers) != N:
        return None
    return centers

def construct():
    """
    Construct 50 circle centers using hexagonal lattice packing with
    various row length patterns, optimized to maximize the common radius.
    """
    
    # Focused set of high-probability patterns
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4], 
        [8, 7, 7, 7, 7, 7, 7, 0], 
        [7, 7, 7, 7, 7, 7, 8, 0], 
        [7, 7, 7, 7, 7, 7, 7, 1],
        [6, 7, 7, 7, 7, 7, 7, 2], 
        [7, 8, 7, 8, 7, 8, 5, 0], 
        [8, 8, 7, 7, 7, 7, 6, 0], 
        [6, 7, 8, 7, 7, 7, 8, 0],
        [7, 7, 7, 7, 7, 6, 7, 2],
        [6, 7, 7, 7, 7, 7, 8, 1],
    ]
    
    valid_patterns = [p for p in patterns if sum(p) == N and len(p) > 0]
    
    best_score = -np.inf
    best_centers = None
    
    for pattern in valid_patterns:
        n_rows = len(pattern)
        max_len = max(pattern)
        sx_init = 1.0 / (max_len + 0.5)
        sy_init = sx_init * np.sqrt(3) / 2.0
        
        total_height = (n_rows - 1) * sy_init
        if total_height > 0.9:
            sy_init = 0.9 / (n_rows - 1) if n_rows > 1 else 0.1
            
        # Fewer, more diverse starts
        starts = [
            [sx_init, sy_init, sx_init / 2.0, 0.0, 0.0],
            [sx_init, sy_init, -sx_init / 2.0, 0.0, 0.0],
        ]
        
        def _neg_score(params):
            centers = _generate_hex_packing(params, pattern)
            if centers is None or centers.shape != (N, 2):
                return 0.0
            return -_score_from_centers(centers)
        
        bounds = [
            (0.01, 0.5),   
            (0.01, 0.5),   
            (-0.5, 0.5),   
            (-0.5, 0.5),   
            (-0.5, 0.5)    
        ]
        
        for x0 in starts:
            try:
                result = minimize(_neg_score, x0, method='Nelder-Mead',
                                  options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10})
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_centers = _generate_hex_packing(result.x, pattern)
            except Exception:
                pass
    
    if best_centers is not None and best_score > 0:
        return best_centers
    
    # Fallback: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
# EVOLVE-BLOCK-END
