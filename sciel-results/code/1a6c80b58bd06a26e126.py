# EVOLVE-BLOCK-START
import numpy as np
import time


def _keich_positions(N):
    """Keich binary construction for N = 2^k base positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _compute_volume(positions, n_samples=500000, seed=12345):
    """
    Compute the Monte Carlo union volume exactly as specified in the task.
    """
    N = 8
    n_tubes = N * N
    x = positions[:, 0]
    y = positions[:, 1]
    i = np.arange(n_tubes) // N
    j = np.arange(n_tubes) % N

    # Box bounds
    X0 = np.min(x)
    X1 = np.max(x + (i + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j + 1) / N)

    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0, None

    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)

    # Check if each sample is inside at least one tube
    # Tube k: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    # Vectorized over tubes for each sample is too expensive (500k * 64), so we loop over tubes
    inside = np.zeros(n_samples, dtype=bool)
    for k in range(n_tubes):
        ix0 = x[k] + c * i[k] / N
        ix1 = ix0 + 1.0 / N
        iy0 = y[k] + c * j[k] / N
        iy1 = iy0 + 1.0 / N
        inside |= (a >= ix0) & (a <= ix1) & (b >= iy0) & (b <= iy1)
        if inside.all():
            break

    fraction = inside.sum() / n_samples
    V = fraction * box_area
    return V, (X0, X1, Y0, Y1)


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    """
    N = 8
    n_tubes = N * N
    deadline = time.time() + 600.0 - 30.0  # 30s safety margin

    # Start with Keich construction
    keich = _keich_positions(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat].copy()
    base_y = keich[j_flat].copy()
    positions = np.stack([base_x, base_y], axis=1)

    # Evaluate initial volume
    V_best, _ = _compute_volume(positions)
    if V_best is None or V_best <= 0 or not np.isfinite(V_best):
        # Fallback: return the Keich positions
        return positions

    # Try to improve using a simple local search / perturbation
    # Since the problem is high-dimensional (128 params), we use coordinate descent
    # with random restarts and small perturbations.
    
    # For efficiency, we'll use a reduced sample count for intermediate evaluations
    n_eval_samples = 100000
    
    best_V = V_best
    best_positions = positions.copy()
    
    rng = np.random.default_rng(42)
    
    # Coordinate descent with random perturbations
    step_size = 0.01
    n_iterations = 0
    max_iterations = 5000  # Limit iterations to stay within time budget
    
    while time.time() < deadline and n_iterations < max_iterations:
        # Pick a random coordinate
        k = rng.integers(0, n_tubes)
        dim = rng.integers(0, 2)  # 0 for x, 1 for y
        
        # Try a small perturbation
        delta = step_size * (2 * rng.random() - 1)
        positions[k, dim] += delta
        
        V, _ = _compute_volume(positions, n_samples=n_eval_samples)
        
        if V is not None and V < best_V and np.isfinite(V):
            best_V = V
            best_positions = positions.copy()
            # Optionally increase step size slightly
            step_size *= 1.1
        else:
            # Revert
            positions[k, dim] -= delta
            step_size *= 0.9
        
        n_iterations += 1
    
    # Final evaluation with full sample count
    V_final, _ = _compute_volume(best_positions, n_samples=500000)
    if V_final is not None and V_final > 0 and np.isfinite(V_final):
        return best_positions
    else:
        return positions


# EVOLVE-BLOCK-END
