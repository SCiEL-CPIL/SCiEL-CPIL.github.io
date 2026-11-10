# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _estimate_volume_fast(params_flat):
    """
    Estimate the union volume using a grid-based Monte Carlo method.
    params_flat: (128,) array of base positions (x_0, y_0, x_1, y_1, ...).
    """
    params = params_flat.reshape(64, 2)
    x = params[:, 0]
    y = params[:, 1]
    i = np.arange(64) // 8
    j = np.arange(64) % 8
    
    c_samples = np.linspace(0, 1, 20)
    total_area = 0.0
    
    for c in c_samples:
        xs = x + c * i / 8.0
        ys = y + c * j / 8.0
        
        x_min = np.min(xs)
        x_max = np.max(xs + 1/8.0)
        y_min = np.min(ys)
        y_max = np.max(ys + 1/8.0)
        
        w = x_max - x_min
        h = y_max - y_min
        
        if w <= 1e-9 or h <= 1e-9:
            continue
            
        # Use a 40x40 grid for area estimation
        grid_size = 40
        gx = np.linspace(x_min + 1e-12, x_max - 1e-12, grid_size)
        gy = np.linspace(y_min + 1e-12, y_max - 1e-12, grid_size)
        gx, gy = np.meshgrid(gx, gy)
        gx_flat = gx.flatten()
        gy_flat = gy.flatten()
        
        mid_xs = xs + 1/16.0
        mid_ys = ys + 1/16.0
        
        # Vectorized check: (N_grid, N_tubes)
        dx = np.abs(gx_flat[:, None] - mid_xs[None, :])
        dy = np.abs(gy_flat[:, None] - mid_ys[None, :])
        
        inside = (dx <= 1/16.0) & (dy <= 1/16.0)
        covered = np.any(inside, axis=1)
        
        area_frac = np.sum(covered) / len(covered)
        total_area += area_frac * w * h
        
    return total_area / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimize 3D Kakeya set using Differential Evolution on 2D base positions.
    """
    # Bounds for x and y coordinates
    bounds = [(-1.0, 2.0)] * 128
    
    # Run Differential Evolution
    result = differential_evolution(
        _estimate_volume_fast, 
        bounds, 
        maxiter=30, 
        popsize=20, 
        seed=42, 
        polish=False, 
        tol=1e-5
    )
    
    best_params = result.x
    
    # Polish with L-BFGS-B
    try:
        res_polish = minimize(
            _estimate_volume_fast, 
            best_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-10}
        )
        if res_polish.fun < result.fun:
            best_params = res_polish.x
    except Exception:
        pass
        
    # Reshape to (64, 2)
    positions = best_params.reshape(64, 2)
    
    # Verify finiteness
    if not np.all(np.isfinite(positions)):
        i_grid, j_grid = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        x = i_grid.flatten() * 0.1
        y = j_grid.flatten() * 0.1
        positions = np.stack([x, y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
