import numpy as np
import time
from scipy.optimize import minimize

def _exact_volume(x, y, N=8):
    """
    Compute the exact union volume of 64 sheared tubes.
    
    Tube (i,j) at height c is the square:
      [x_i + c*i/N, x_i + c*i/N + 1/N] x [y_j + c*j/N, y_j + c*j/N + 1/N]
    
    The union volume is the integral over c in [0,1] of the area of the union of
    the 64 squares at height c.
    
    We use a grid of c values to compute this integral accurately.
    """
    w = 1.0 / N
    n_c = 200  # number of c samples for integration
    c_vals = np.linspace(0, 1, n_c)
    
    total_area = 0.0
    for c in c_vals:
        # Compute x-intervals for all i
        x_ls = x + c * np.arange(N) / N
        x_rs = x_ls + w
        
        # Compute y-intervals for all j
        y_ls = y + c * np.arange(N) / N
        y_rs = y_ls + w
        
        # For each pair (i,j), we have a square [x_ls[i], x_rs[i]] x [y_ls[j], y_rs[j]]
        # We need the area of the union of all 64 such squares.
        
        # Collect all unique x boundaries and y boundaries
        all_x_bounds = np.concatenate([x_ls, x_rs])
        all_y_bounds = np.concatenate([y_ls, y_rs])
        
        # Sort unique boundaries
        x_bounds = np.unique(np.round(all_x_bounds, 12))
        y_bounds = np.unique(np.round(all_y_bounds, 12))
        
        # For each cell in the x-y grid, check if it's covered by at least one square
        # A cell [x_bounds[k], x_bounds[k+1]] x [y_bounds[m], y_bounds[m+1]] is covered
        # if there exists (i,j) such that x_ls[i] <= x_bounds[k] and x_bounds[k+1] <= x_rs[i]
        # and y_ls[j] <= y_bounds[m] and y_bounds[m+1] <= y_rs[j]
        
        area = 0.0
        for k in range(len(x_bounds) - 1):
            xl = x_bounds[k]
            xr = x_bounds[k + 1]
            if xr - xl < 1e-12:
                continue
            # Find which i cover this x-range
            i_cover = [i for i in range(N) if x_ls[i] <= xl + 1e-12 and xr <= x_rs[i] + 1e-12]
            if not i_cover:
                continue
            for m in range(len(y_bounds) - 1):
                yl = y_bounds[m]
                yr = y_bounds[m + 1]
                if yr - yl < 1e-12:
                    continue
                # Find which j cover this y-range
                j_cover = [j for j in range(N) if y_ls[j] <= yl + 1e-12 and yr <= y_rs[j] + 1e-12]
                if not j_cover:
                    continue
                # Cell is covered if there's at least one (i,j) pair
                # Since any i in i_cover and any j in j_cover gives a square covering this cell
                area += (xr - xl) * (yr - yl)
        
        total_area += area
    
    # Trapezoidal integration over c
    volume = np.trapz(total_area / n_c * np.ones_like(c_vals), c_vals)
    # Actually, total_area is already the sum over c, so we need to divide by n_c and integrate
    # Let me redo: total_area should be the sum of areas at each c, then volume = mean * 1
    # Actually, the volume is the integral of area(c) dc, which is approximately mean(area) * 1
    return total_area / n_c

def _exact_volume_fast(x, y, N=8):
    """
    Faster exact volume computation using vectorized operations.
    """
    w = 1.0 / N
    n_c = 100
    c_vals = np.linspace(0, 1, n_c)
    
    areas = np.zeros(n_c)
    
    for idx, c in enumerate(c_vals):
        x_ls = x + c * np.arange(N) / N
        x_rs = x_ls + w
        y_ls = y + c * np.arange(N) / N
        y_rs = y_ls + w
        
        # Build coverage grid
        # x_bounds: unique sorted x boundaries
        x_bounds = np.unique(np.sort(np.concatenate([x_ls, x_rs])))
        y_bounds = np.unique(np.sort(np.concatenate([y_ls, y_rs])))
        
        area = 0.0
        for k in range(len(x_bounds) - 1):
            xl, xr = x_bounds[k], x_bounds[k + 1]
            dx = xr - xl
            if dx < 1e-12:
                continue
            # Which i cover [xl, xr]?
            i_mask = (x_ls <= xl + 1e-12) & (x_rs >= xr - 1e-12)
            if not np.any(i_mask):
                continue
            for m in range(len(y_bounds) - 1):
                yl, yr = y_bounds[m], y_bounds[m + 1]
                dy = yr - yl
                if dy < 1e-12:
                    continue
                j_mask = (y_ls <= yl + 1e-12) & (y_rs >= yr - 1e-12)
                if not np.any(j_mask):
                    continue
                area += dx * dy
        
        areas[idx] = area
    
    return np.mean(areas)

def _objective(params, N=8):
    """
    Objective: minimize the union volume.
    params: 2N array, first N are x positions, last N are y positions.
    """
    x = params[:N]
    y = params[N:]
    
    # Compute exact volume
    vol = _exact_volume_fast(x, y, N)
    
    # Add a small penalty to keep things in a reasonable range
    # This helps the optimizer not go to infinity
    spread_x = np.max(x) - np.min(x)
    spread_y = np.max(y) - np.min(y)
    penalty = 0.01 * (spread_x + spread_y)
    
    return vol + penalty

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    N = 8
    w = 1.0 / N
    
    # Initial guess: spread positions to minimize overlap
    # A good initial guess is to have x_i and y_j spread out
    x0 = np.linspace(-0.5, 0.5, N)
    y0 = np.linspace(-0.5, 0.5, N)
    
    best_params = np.concatenate([x0, y0])
    best_val = _objective(best_params, N)
    
    # Multiple restarts with L-BFGS-B
    rng = np.random.default_rng(12345)
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params = best_params.copy()
        else:
            # Random initialization
            scale = 1.0
            params = rng.uniform(-scale, scale, 2 * N)
        
        try:
            res = minimize(_objective, params, args=(N,), method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Also try Powell for robustness
    if time.time() < deadline - 60:
        try:
            res = minimize(_objective, best_params, args=(N,), method='Powell',
                          options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    # Construct the 64x2 array
    # Row k is tube (i, j) with i = k // 8 and j = k % 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
