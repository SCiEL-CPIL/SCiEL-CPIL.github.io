import numpy as np
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Minimize the union volume of 64 sheared tubes using optimization.
    """
    N = 8
    n_tubes = N * N  # 64 tubes
    
    def compute_volume(positions, n_samples=500000, seed=12345):
        """Compute union volume via Monte Carlo."""
        positions = np.asarray(positions).reshape(n_tubes, 2)
        i_idx = np.repeat(np.arange(N), N)
        j_idx = np.tile(np.arange(N), N)
        
        x_base = positions[:, 0]
        y_base = positions[:, 1]
        
        # Compute bounding box
        x_min = np.min(x_base)
        x_max = np.max(x_base + (i_idx + 1) / N)
        y_min = np.min(y_base)
        y_max = np.max(y_base + (j_idx + 1) / N)
        
        if (x_max - x_min) * (y_max - y_min) <= 1e-9:
            return 1e9  # Invalid
        
        rng = np.random.default_rng(seed)
        a = rng.uniform(x_min, x_max, n_samples)
        b = rng.uniform(y_min, y_max, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        # Check if point (a, b, c) is inside any tube
        # Tube k at height c: x in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
        #                    y in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
        
        # Vectorized check
        x_left = x_base[None, :] + c[:, None] * i_idx[None, :] / N
        x_right = x_left + 1.0 / N
        y_left = y_base[None, :] + c[:, None] * j_idx[None, :] / N
        y_right = y_left + 1.0 / N
        
        inside_x = (a[:, None] >= x_left) & (a[:, None] <= x_right)
        inside_y = (b[:, None] >= y_left) & (b[:, None] <= y_right)
        inside = inside_x & inside_y
        
        fraction_inside = np.any(inside, axis=1).mean()
        volume = fraction_inside * (x_max - x_min) * (y_max - y_min)
        
        if not np.isfinite(volume) or volume <= 0:
            return 1e9
        return volume
    
    # Initial guess: structured arrangement
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Start with a diagonal arrangement
    x_init = (i_flat + j_flat) / (2.0 * N)
    y_init = (i_flat - j_flat) / (2.0 * N) + 0.5
    
    positions_init = np.stack([x_init, y_init], axis=1).flatten()
    
    # Use Nelder-Mead optimization
    result = minimize(
        compute_volume,
        positions_init,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False}
    )
    
    positions_opt = result.x.reshape(n_tubes, 2)
    
    # Validate
    if not np.all(np.isfinite(positions_opt)):
        # Fall back to initial guess
        positions_opt = positions_init.reshape(n_tubes, 2)
    
    return positions_opt
