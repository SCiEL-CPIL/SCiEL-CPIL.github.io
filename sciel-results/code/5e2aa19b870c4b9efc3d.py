# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _estimate_volume(x, y, N, n_samples=50000, seed=42):
    """
    Estimate the 3D union volume using Monte Carlo.
    x: array of shape (N,) base positions for i
    y: array of shape (N,) base positions for j
    """
    rng = np.random.default_rng(seed)
    
    # Calculate bounding box
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    
    # Max x extent: max_k (x_k + (i+1)/8) where k=(i,j)
    # x_k depends only on i. So max over i of x_i + (i+1)/8
    x_max = np.max(x + (i_vals + 1) / N)
    x_min = np.min(x)
    
    y_max = np.max(y + (j_vals + 1) / N)
    y_min = np.min(y)
    
    if x_max - x_min <= 1e-9 or y_max - y_min <= 1e-9:
        return 1e9 # Large penalty for degenerate box
        
    a = rng.uniform(x_min, x_max, n_samples)
    b = rng.uniform(y_min, y_max, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Prepare tube indices
    i_flat, j_flat = np.meshgrid(i_vals, j_vals, indexing="ij")
    i_flat = i_flat.flatten()
    j_flat = j_flat.flatten()
    
    # Vectorized check
    # c shape (S, 1)
    c_exp = c[:, None]
    a_exp = a[:, None]
    b_exp = b[:, None]
    
    # i_flat shape (1, 64)
    i_exp = i_flat[None, :]
    j_exp = j_flat[None, :]
    
    # x shape (1, N) -> broadcast to (1, 64) via i_exp
    # x[i] + c * i / N
    x_starts = x[i_exp] + c_exp * (i_exp / N)
    x_ends = x_starts + 1.0 / N
    
    y_starts = y[j_exp] + c_exp * (j_exp / N)
    y_ends = y_starts + 1.0 / N
    
    inside = (a_exp >= x_starts) & (a_exp <= x_ends) & (b_exp >= y_starts) & (b_exp <= y_ends)
    
    hit = inside.any(axis=1)
    vol = np.mean(hit) * (x_max - x_min) * (y_max - y_min)
    
    return vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with N = 8.
    Optimizes x and y base positions jointly to minimize 3D union volume.
    """
    N = 8
    
    # Initial guess: folding pattern
    def initial_guess():
        pos = np.zeros(N)
        for i in range(N):
            pos[i] = 1.0 / (2 * N) - i / N + (1.0 / (2 * N)) * (i % 2)
        pos -= pos.min()
        return pos
    
    x0 = initial_guess()
    y0 = initial_guess()
    
    params0 = np.concatenate([x0, y0])
    
    def objective(params):
        x = params[:N]
        y = params[N:]
        return _estimate_volume(x, y, N, n_samples=20000, seed=42)
        
    # Use Nelder-Mead for non-smooth (MC noise) optimization
    # Or use COBYLA? Nelder-Mead is often robust for low-dim.
    # 16 dimensions.
    
    # To save time, we can do a few iterations of Nelder-Mead
    # Or just use a simple gradient descent with finite differences if we smooth the noise?
    # MC noise is high. Nelder-Mead handles non-smoothness better.
    
    # Limit iterations to avoid timeout
    res = minimize(objective, params0, method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5})
    
    best_params = res.x
    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    # Final polish with higher sample size?
    # Let's just return the optimized positions.
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
