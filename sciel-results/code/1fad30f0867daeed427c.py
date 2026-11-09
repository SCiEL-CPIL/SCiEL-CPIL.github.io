# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_BUDGET = 630.0
_MARGIN = 45.0


def _fast_volume(positions, seed=42, n_samples=20000):
    """Fast Monte Carlo volume estimate for optimization."""
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N

    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)

    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')

    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)

    # Vectorized: for each tube k, check if point (a,b,c) is inside
    # Tube k: x in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
    #         y in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    # Shape: (64, n_samples)
    x_left = x[:, None] + c[None, :] * i_idx[:, None] / N
    x_right = x_left + 1.0 / N
    y_left = y[:, None] + c[None, :] * j_idx[:, None] / N
    y_right = y_left + 1.0 / N

    in_x = (a[None, :] >= x_left) & (a[None, :] <= x_right)
    in_y = (b[None, :] >= y_left) & (b[None, :] <= y_right)

    inside = np.any(in_x & in_y, axis=0)
    fraction = np.mean(inside)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V


def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _build_positions(px, py, N):
    px = np.asarray(px, dtype=np.float64).ravel()[:N]
    py = np.asarray(py, dtype=np.float64).ravel()[:N]
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = px[i_flat]
    base_y = py[j_flat]
    return np.stack([base_x, base_y], axis=1)


def _local_search(positions, max_time, n_samples=15000, seed=42):
    """Direct local search on fast Monte Carlo volume."""
    start_time = time.time()
    current = positions.copy()
    current_V = _fast_volume(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V

    rng = np.random.default_rng(seed)
    step = 0.1

    while time.time() - start_time < max_time:
        improved = False
        indices = rng.permutation(64)

        for k in indices:
            if time.time() - start_time > max_time:
                break
            for coord in [0, 1]:
                for direction in [1, -1]:
                    trial = current.copy()
                    trial[k, coord] += direction * step * rng.uniform(0.5, 1.5)
                    trial_V = _fast_volume(trial, n_samples=n_samples, seed=seed)
                    if trial_V < current_V - 1e-12:
                        current = trial
                        current_V = trial_V
                        improved = True
                        if current_V < best_V:
                            best = current.copy()
                            best_V = current_V

        if not improved:
            step *= 0.7
            if step < 0.001:
                break

    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich(N)
    px0 = keich.copy()
    py0 = keich.copy()
    P0 = _build_positions(px0, py0, N)

    # Start with Keich construction
    P = P0.copy()

    # Deadline guard
    deadline = _T0 + _BUDGET - _MARGIN

    # Local search with fast Monte Carlo objective
    try:
        remaining = deadline - time.time()
        if remaining > 30:
            P_refined, _ = _local_search(P, max_time=remaining - 10, n_samples=15000, seed=42)
            if np.all(np.isfinite(P_refined)):
                P = P_refined
    except Exception:
        pass

    # Final sanity check
    if not np.all(np.isfinite(P)):
        P = P0

    return P


# EVOLVE-BLOCK-END
