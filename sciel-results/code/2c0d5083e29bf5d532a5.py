# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume(positions, seed=12345, n_samples=500000):
    """Compute the union volume using the exact Monte Carlo method from the task."""
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]

    # Get i and j indices for each tube
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N

    # Compute box bounds
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

    # Check if each point is inside at least one tube
    # Tube k at height c: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    inside = np.zeros(n_samples, dtype=bool)

    for k in range(64):
        i_k = i_idx[k]
        j_k = j_idx[k]
        xk = x[k]
        yk = y[k]

        x_left = xk + c * i_k / N
        x_right = x_left + 1.0 / N
        y_bottom = yk + c * j_k / N
        y_top = y_bottom + 1.0 / N

        mask = (a >= x_left) & (a <= x_right) & (b >= y_bottom) & (b <= y_top)
        inside |= mask

    fraction = np.mean(inside)
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    return volume


def _local_search(positions, max_iter=500, step_size=0.01, deadline=None):
    """Perform local search to minimize the union volume."""
    current_pos = positions.copy()
    current_vol = _compute_volume(current_pos)
    best_pos = current_pos.copy()
    best_vol = current_vol

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Try random perturbations
        improved = False
        for _ in range(10):
            trial_pos = current_pos.copy()
            # Perturb a few random positions
            indices = np.random.choice(64, size=4, replace=False)
            for idx in indices:
                trial_pos[idx, 0] += np.random.uniform(-step_size, step_size)
                trial_pos[idx, 1] += np.random.uniform(-step_size, step_size)

            trial_vol = _compute_volume(trial_pos)
            if trial_vol < current_vol:
                current_pos = trial_pos
                current_vol = trial_vol
                improved = True
                if current_vol < best_vol:
                    best_pos = current_pos.copy()
                    best_vol = current_vol

        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break

    return best_pos, best_vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].
    So the tube is a unit-height prism of square cross-section 1/N x 1/N that is
    sheared so its top face is offset by (i/N, j/N): the (i, j) pair selects the
    needle direction, and all N*N directions are covered. The task is to choose the
    64 base positions so that the VOLUME of the union of all tubes is as SMALL as
    possible (a Kakeya/Besicovitch-type set: needles in every direction, tiny measure).

    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline

    N = 8
    # Keich binary construction for n = 2^k base positions (one per grid index).
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
    positions = np.stack([base_x, base_y], axis=1)

    # Perform local search to improve the initial position
    best_pos, best_vol = _local_search(positions, max_iter=200, step_size=0.05, deadline=deadline)

    return best_pos


# EVOLVE-BLOCK-END
