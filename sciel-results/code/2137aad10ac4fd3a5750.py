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
    
    inside = np.zeros(n_samples, dtype=bool)
    for k in range(64):
        i_k = i_flat[k]
        j_k = j_flat[k]
        xk = positions[k, 0]
        yk = positions[k, 1]
        x_start = xk + c * i_k / N
        x_end = x_start + 1.0 / N
        y_start = yk + c * j_k / N
        y_end = y_start + 1.0 / N
        in_tube = (a >= x_start) & (a <= x_end) & (b >= y_start) & (b <= y_end)
        inside |= in_tube
    
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


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 590
    
    candidates = [
        _keich_positions(N),
        _fan_positions(N),
        _origin_positions(N),
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
