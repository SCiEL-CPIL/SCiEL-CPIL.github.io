import numpy as np
import time

def _compute_volume(positions, n_samples=500000, seed=12345):
    """Compute the union volume using the specified Monte Carlo method."""
    N = 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    
    # Compute bounding box
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_idx + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_idx + 1) / N)
    
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return 0.0
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is in at least one tube
    # Tube (i,j) at height c: x in [x_k + c*i/N, x_k + c*i/N + 1/N], y in [y_k + c*j/N, y_k + c*j/N + 1/N]
    # For all 64 tubes, check if any contains the point
    
    # Vectorize: for each sample point, check all 64 tubes
    # a has shape (n_samples,), positions has shape (64, 2)
    # x_k + c*i/N: shape (64,) * c[:,None] * i_idx[None,:] -> (n_samples, 64)
    
    x_shift = c[:, None] * i_idx[None, :] / N  # (n_samples, 64)
    y_shift = c[:, None] * j_idx[None, :] / N  # (n_samples, 64)
    
    x_left = x_k[None, :] + x_shift  # (n_samples, 64)
    x_right = x_left + 1.0 / N
    y_left = y_k[None, :] + y_shift  # (n_samples, 64)
    y_right = y_left + 1.0 / N
    
    in_x = (a[:, None] >= x_left) & (a[:, None] <= x_right)  # (n_samples, 64)
    in_y = (b[:, None] >= y_left) & (b[:, None] <= y_right)  # (n_samples, 64)
    
    in_any = np.any(in_x & in_y, axis=1)
    
    fraction = np.mean(in_any)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    
    return V


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Start from the current best solution
    base_x = -np.arange(N) / (2 * N)
    base_y = -np.arange(N) / (2 * N)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x0 = base_x[i_flat].copy()
    y0 = base_y[j_flat].copy()
    
    positions = np.stack([x0, y0], axis=1)
    
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Local search using coordinate descent on individual positions
    best_positions = positions.copy()
    best_volume = _compute_volume(best_positions, n_samples=200000, seed=12345)
    
    step_size = 0.05
    n_iter = 0
    
    while time.time() < deadline:
        improved = False
        for k in range(64):
            for coord in range(2):
                # Try positive and negative steps
                for sign in [1, -1]:
                    new_pos = best_positions.copy()
                    new_pos[k, coord] += sign * step_size
                    vol = _compute_volume(new_pos, n_samples=200000, seed=12345)
                    if vol < best_volume:
                        best_positions = new_pos
                        best_volume = vol
                        improved = True
                        break
            if time.time() > deadline:
                break
        if not improved:
            step_size *= 0.7
            if step_size < 0.001:
                break
        n_iter += 1
    
    return best_positions
