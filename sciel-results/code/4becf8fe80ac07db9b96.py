# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume(positions, n_samples=100000, seed=12345):
    """
    Compute the Monte Carlo estimate of the union volume.
    
    Args:
        positions: array of shape (64, 2) with base positions (x_k, y_k)
        n_samples: number of Monte Carlo samples
        seed: random seed for reproducibility
    
    Returns:
        Volume estimate
    """
    N = 8
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    
    # Extract base positions
    base_x = positions[:, 0]
    base_y = positions[:, 1]
    
    # Compute X0, X1, Y0, Y1
    x_offsets = np.zeros(64)
    y_offsets = np.zeros(64)
    for k in range(64):
        i = k // N
        j = k % N
        x_offsets[k] = (i + 1) / N
        y_offsets[k] = (j + 1) / N
    
    X0 = np.min(base_x)
    X1 = np.max(base_x + x_offsets)
    Y0 = np.min(base_y)
    Y1 = np.max(base_y + y_offsets)
    
    # Check validity
    if not np.all(np.isfinite([X0, X1, Y0, Y1])):
        return np.inf
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return np.inf
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if each point is inside at least one tube
    # Vectorized: for each tube, check if point is inside
    inside_any = np.zeros(n_samples, dtype=bool)
    
    # Process in chunks to manage memory
    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        a_chunk = a[start:end]
        b_chunk = b[start:end]
        c_chunk = c[start:end]
        
        for k in range(64):
            i = k // N
            j = k % N
            x_lo = base_x[k] + c_chunk * i / N
            x_hi = x_lo + 1 / N
            y_lo = base_y[k] + c_chunk * j / N
            y_hi = y_lo + 1 / N
            
            inside_k = (a_chunk >= x_lo) & (a_chunk <= x_hi) & \
                       (b_chunk >= y_lo) & (b_chunk <= y_hi)
            
            inside_any[start:end] |= inside_k
    
    fraction = np.mean(inside_any)
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    
    return volume


def _keich_construction(N=8):
    """Generate the Keich binary construction baseline."""
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
    return positions


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses simulated annealing to minimize the union volume of 64 sheared tubes.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    deadline = time.time() + 580  # Safety margin before 600s limit
    
    # Start from Keich construction
    best_positions = _keich_construction(N)
    best_volume = _compute_volume(best_positions, n_samples=50000, seed=12345)
    
    # Simulated annealing parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    max_steps = 5000
    
    current_positions = best_positions.copy()
    current_volume = best_volume
    
    rng = np.random.default_rng(42)
    
    step = 0
    while step < max_steps and time.time() < deadline:
        step += 1
        
        # Generate a random perturbation
        # Perturb a random subset of positions
        n_perturb = rng.integers(1, 8)
        indices = rng.choice(64, size=n_perturb, replace=False)
        
        # Random direction and magnitude
        magnitude = T * 0.1
        perturbation = rng.normal(0, magnitude, size=(n_perturb, 2))
        
        new_positions = current_positions.copy()
        new_positions[indices] += perturbation
        
        # Evaluate new volume (use fewer samples for speed during optimization)
        new_volume = _compute_volume(new_positions, n_samples=20000, seed=12345 + step)
        
        if not np.isfinite(new_volume):
            new_volume = np.inf
        
        # Accept or reject
        if new_volume < current_volume:
            current_positions = new_positions
            current_volume = new_volume
            if current_volume < best_volume:
                best_positions = current_positions.copy()
                best_volume = current_volume
        else:
            # Metropolis criterion
            delta = new_volume - current_volume
            if delta < 0:
                current_positions = new_positions
                current_volume = new_volume
            else:
                accept_prob = np.exp(-delta / T)
                if rng.random() < accept_prob:
                    current_positions = new_positions
                    current_volume = new_volume
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # Periodic restarts from best solution if stuck
        if step % 500 == 0 and current_volume > best_volume * 1.1:
            current_positions = best_positions.copy()
            current_volume = best_volume
            T = max(T * 10, 0.01)
    
    # Final verification with full sample count
    final_volume = _compute_volume(best_positions, n_samples=500000, seed=12345)
    if not np.isfinite(final_volume) or final_volume <= 0:
        # Fall back to Keich construction
        return _keich_construction(N)
    
    return best_positions


# EVOLVE-BLOCK-END
