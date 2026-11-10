# EVOLVE-BLOCK-START
import numpy as np
import time


def _mc_volume(positions, N=8, n_samples=500000, seed=12345):
    """Compute the Monte Carlo volume estimate for given positions."""
    rng = np.random.default_rng(seed)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x_starts = positions[:, 0]
    x_ends = positions[:, 0] + (i_flat + 1) / N
    y_starts = positions[:, 1]
    y_ends = positions[:, 1] + (j_flat + 1) / N
    
    X0 = np.min(x_starts)
    X1 = np.max(x_ends)
    Y0 = np.min(y_starts)
    Y1 = np.max(y_ends)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Vectorized: for each tube k, check if point is inside
    # Tube k: x in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
    #         y in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    # Shape: (64, n_samples)
    x_lo = positions[:, 0:1] + c[None, :] * i_flat[:, None] / N
    x_hi = x_lo + 1.0 / N
    y_lo = positions[:, 1:2] + c[None, :] * j_flat[:, None] / N
    y_hi = y_lo + 1.0 / N
    
    in_x = (a[None, :] >= x_lo) & (a[None, :] <= x_hi)
    in_y = (b[None, :] >= y_lo) & (b[None, :] <= y_hi)
    inside = np.any(in_x & in_y, axis=0)
    
    fraction = np.sum(inside) / n_samples
    return fraction * (X1 - X0) * (Y1 - Y0)


def _keich_positions(N=8):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    return np.stack([base_x, base_y], axis=1)


def _fan_positions(N=8):
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = -i_flat / N
    base_y = -j_flat / N
    return np.stack([base_x, base_y], axis=1)


def _origin_positions(N=8):
    return np.zeros((N*N, 2))


def _diagonal_positions(N=8):
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    # Place tubes along a diagonal to minimize overlap
    base_x = -i_flat / N + 0.5 * j_flat / N
    base_y = -j_flat / N + 0.5 * i_flat / N
    return np.stack([base_x, base_y], axis=1)


def _compact_positions(N=8):
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    # Try to pack tubes into a small region
    # At c=0: tube (i,j) is at [x_k, x_k+1/N] x [y_k, y_k+1/N]
    # At c=1: tube (i,j) is at [x_k+i/N, x_k+i/N+1/N] x [y_k+j/N, y_k+j/N+1/N]
    # To minimize union, we want all tubes to overlap as much as possible
    # If we set x_k = -i/N * alpha and y_k = -j/N * alpha for some alpha in [0,1]
    # At c: x range is [-i/N*alpha + c*i/N, -i/N*alpha + c*i/N + 1/N]
    #      = [i/N*(c-alpha), i/N*(c-alpha) + 1/N]
    # At c=0: [i/N*(-alpha), i/N*(-alpha)+1/N]
    # At c=1: [i/N*(1-alpha), i/N*(1-alpha)+1/N]
    # The union in x for all i is from min_i(i/N*(-alpha)) to max_i(i/N*(1-alpha)+1/N)
    # = from 0 (if alpha>=0) to (N-1)/N*(1-alpha)+1/N
    # To minimize, set alpha=1: then at c=0: [-i/N, -i/N+1/N], at c=1: [0, 1/N]
    # Union x: from -7/8 to 1/8, width = 1
    # Same for y
    # So compact with alpha=1 gives a 1x1x1 box, volume = fraction * 1
    base_x = -i_flat / N
    base_y = -j_flat / N
    return np.stack([base_x, base_y], axis=1)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 590
    
    candidates = [
        _keich_positions(N),
        _fan_positions(N),
        _origin_positions(N),
        _diagonal_positions(N),
        _compact_positions(N),
    ]
    
    best_pos = candidates[0]
    best_vol = float('inf')
    
    for pos in candidates:
        if time.time() > deadline:
            break
        vol = _mc_volume(pos, N)
        if vol < best_vol:
            best_vol = vol
            best_pos = pos
    
    return best_pos


# EVOLVE-BLOCK-END
