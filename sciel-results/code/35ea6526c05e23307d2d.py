# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_1d_union_length(base_positions, c_values, N=8):
    """
    Compute the union length of intervals [x_i + c*i/N, x_i + 1/N + c*i/N]
    at given height values c.
    
    Args:
        base_positions: array of shape (N,) with base x positions
        c_values: array of height values
        N: grid parameter (default 8)
    
    Returns:
        array of union lengths at each c value
    """
    N = len(base_positions)
    c_values = np.atleast_1d(c_values)
    num_c = len(c_values)
    
    # For each c, compute all interval endpoints
    # L_i(c) = x_i + (i/N) * c
    # R_i(c) = x_i + 1/N + (i/N) * c
    
    # Create meshgrid for efficient computation
    indices = np.arange(N)
    # Shape: (num_c, N)
    left = base_positions[None, :] + (indices[None, :] / N) * c_values[:, None]
    right = left + 1.0 / N
    
    # Compute union length by sorting endpoints
    union_lengths = np.zeros(num_c)
    for ic in range(num_c):
        # Sort all endpoints
        all_ends = np.sort(np.concatenate([left[ic], right[ic]]))
        # Sweep line to compute union length
        length = 0.0
        current_end = -np.inf
        for k in range(0, len(all_ends), 2):
            start = all_ends[k]
            end = all_ends[k + 1]
            if start > current_end:
                length += end - start
                current_end = end
            else:
                current_end = max(current_end, end)
        union_lengths[ic] = length
    
    return union_lengths


def _compute_1d_volume(base_positions, num_samples=10000, N=8):
    """
    Estimate the 1D union volume (integral of union length over c in [0,1]).
    
    Args:
        base_positions: array of shape (N,)
        num_samples: number of Monte Carlo samples
        N: grid parameter
    
    Returns:
        estimated volume
    """
    rng = np.random.default_rng(42)
    c_samples = rng.uniform(0, 1, num_samples)
    lengths = _compute_1d_union_length(base_positions, c_samples, N)
    return np.mean(lengths)


def _optimize_1d_positions(initial_positions, max_iter=50, N=8):
    """
    Optimize 1D base positions to minimize union volume.
    
    Args:
        initial_positions: initial array of shape (N,)
        max_iter: maximum iterations
        N: grid parameter
    
    Returns:
        optimized positions
    """
    def objective(x):
        return _compute_1d_volume(x, num_samples=2000, N=N)
    
    result = minimize(objective, initial_positions, method='Nelder-Mead',
                     options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
    return result.x


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses numerical optimization to find base positions that minimize the union volume.
    Exploits the product structure: optimize x-positions (function of i) and y-positions
    (function of j) independently.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 570  # 570 seconds to leave safety margin
    
    # Start with Keich construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Normalize to [0, 1] range for better optimization
    keich_norm = (keich - keich.min()) / (keich.max() - keich.min() + 1e-10)
    
    # Optimize x-positions
    best_x = keich_norm.copy()
    best_x_vol = _compute_1d_volume(best_x, num_samples=5000, N=N)
    
    # Try multiple random initializations
    rng = np.random.default_rng(12345)
    for trial in range(3):
        if time.time() > deadline:
            break
        init = rng.uniform(0, 1, N)
        try:
            opt_x = _optimize_1d_positions(init, max_iter=30, N=N)
            vol_x = _compute_1d_volume(opt_x, num_samples=5000, N=N)
            if vol_x < best_x_vol:
                best_x = opt_x
                best_x_vol = vol_x
        except Exception:
            continue
    
    # Optimize y-positions (same optimization, different seed)
    best_y = keich_norm.copy()
    best_y_vol = _compute_1d_volume(best_y, num_samples=5000, N=N)
    
    rng2 = np.random.default_rng(54321)
    for trial in range(3):
        if time.time() > deadline:
            break
        init = rng2.uniform(0, 1, N)
        try:
            opt_y = _optimize_1d_positions(init, max_iter=30, N=N)
            vol_y = _compute_1d_volume(opt_y, num_samples=5000, N=N)
            if vol_y < best_y_vol:
                best_y = opt_y
                best_y_vol = vol_y
        except Exception:
            continue
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
