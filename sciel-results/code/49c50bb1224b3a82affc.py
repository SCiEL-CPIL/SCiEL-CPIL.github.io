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
        x_k = x[k]
        y_k = y[k]
        
        x_left = x_k + c * i_k / N
        x_right = x_left + 1.0 / N
        y_left = y_k + c * j_k / N
        y_right = y_left + 1.0 / N
        
        in_x = (a >= x_left) & (a <= x_right)
        in_y = (b >= y_left) & (b <= y_right)
        inside |= (in_x & in_y)
    
    fraction = np.mean(inside)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V


def _local_search(positions, max_time=580, n_samples=50000):
    """Local search with adaptive step sizing, multiple restarts, and final polish."""
    start_time = time.time()
    best = positions.copy()
    best_V = _compute_volume(best, n_samples=n_samples)
    
    N = 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    rng = np.random.default_rng(42)
    
    def _coordinate_descent(current, current_V, max_steps, step_sizes, time_budget):
        """Run coordinate descent with given step sizes."""
        improved = True
        step_idx = 0
        while improved and step_idx < len(step_sizes) and time.time() - start_time < time_budget:
            step = step_sizes[step_idx]
            improved = False
            for _ in range(max_steps):
                if time.time() - start_time > time_budget:
                    break
                for k in range(64):
                    for coord in range(2):
                        for direction in [-1, 1]:
                            trial = current.copy()
                            trial[k, coord] += direction * step * rng.uniform(0.5, 1.5)
                            if trial[k, coord] < -1 or trial[k, coord] > 2:
                                continue
                            trial_V = _compute_volume(trial, n_samples=n_samples)
                            if trial_V < current_V - 1e-10:
                                current = trial
                                current_V = trial_V
                                improved = True
            if improved:
                step_idx += 1
        return current, current_V
    
    # Phase 1: Multiple restarts with perturbations
    n_restarts = 5
    restart_budget = max_time * 0.6
    for r in range(n_restarts):
        if time.time() - start_time > restart_budget + 30:
            break
        # Create perturbed start
        perturbation_scale = 0.1 * (1 - r / n_restarts) + 0.02
        current = best + rng.normal(0, perturbation_scale, size=best.shape)
        current = np.clip(current, -1, 2)
        current_V = _compute_volume(current, n_samples=n_samples)
        
        # Run coordinate descent
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        current, current_V = _coordinate_descent(
            current, current_V, max_steps=3, step_sizes=step_sizes,
            time_budget=restart_budget + 30
        )
        
        if current_V < best_V:
            best = current.copy()
            best_V = current_V
    
    # Phase 2: Final high-precision polish pass
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 60:
        current = best.copy()
        current_V = best_V
        polish_step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
        current, current_V = _coordinate_descent(
            current, current_V, max_steps=5, step_sizes=polish_step_sizes,
            time_budget=max_time - 30
        )
        if current_V < best_V:
            best = current.copy()
            best_V = current_V
    
    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then refines with local search.
    """
    N = 8
    
    # Keich binary construction for n = 2^k base positions
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
    
    # Deadline guard
    start_time = time.time()
    max_total_time = 620  # Leave 10s margin
    
    # Try local search if we have time
    if time.time() - start_time < max_total_time - 100:
        try:
            refined, _ = _local_search(positions, max_time=max_total_time - 20)
            positions = refined
        except Exception:
            pass  # Keep original if search fails
    
    return positions


# EVOLVE-BLOCK-END
