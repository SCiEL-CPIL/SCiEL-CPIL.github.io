"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _cell_params(grid_x, grid_y):
    """
    Calculate centers and inradii for all cells.
    Returns:
        centers: (N_CELLS, 2) array of [x, y]
        radii: (N_CELLS,) array of inradii
    """
    nx = len(grid_x) - 1
    ny = len(grid_y) - 1
    n_cells = nx * ny
    centers = np.zeros((n_cells, 2))
    radii = np.zeros(n_cells)
    
    idx = 0
    for r in range(ny):
        for c in range(nx):
            x_left = grid_x[c]
            x_right = grid_x[c + 1]
            y_bottom = grid_y[r]
            y_top = grid_y[r + 1]
            
            cx = (x_left + x_right) / 2.0
            cy = (y_bottom + y_top) / 2.0
            
            # Inradius is half the min dimension
            w_cell = x_right - x_left
            h_cell = y_top - y_bottom
            r_cell = min(w_cell, h_cell) / 2.0
            
            centers[idx] = [cx, cy]
            radii[idx] = r_cell
            idx += 1
            
    return centers, radii

def _objective_grid(z, w, h, nx, ny, active_mask):
    """
    Objective function for grid optimization.
    z layout: [gx_0, ..., gx_{nx-2}, gy_0, ..., gy_{ny-2}]
    active_mask: boolean array of length nx*ny
    """
    gx = z[0:nx-1]
    gy = z[nx-1:nx-1+ny-1]
    
    # Ensure grid lines are within bounds and sorted
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
        
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    # Check if any cell is too small
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e6
        
    centers, radii = _cell_params(grid_x, grid_y)
    
    active_idx = np.where(active_mask)[0]
    active_centers = centers[active_idx]
    active_radii = radii[active_idx]
    
    total_r = np.sum(active_radii)
    
    # Overlap penalty between active circles
    n_active = len(active_idx)
    if n_active > 1:
        dx = active_centers[:, 0, None] - active_centers[None, :, 0]
        dy = active_centers[:, 1, None] - active_centers[None, :, 1]
        d_sq = dx * dx + dy * dy
        r_sum = active_radii[:, None] + active_radii[None, :]
        
        iu = np.triu_indices(n_active, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        r_sum_pairs = r_sum[iu]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        penalty = 1000.0 * np.sum(overlap * overlap)
    else:
        penalty = 0.0

    # Penalty for inactive cells having significant size
    inactive_idx = np.where(~active_mask)[0]
    if len(inactive_idx) > 0:
        inactive_radii = radii[inactive_idx]
        inactive_penalty = 10.0 * np.sum(inactive_radii)
    else:
        inactive_penalty = 0.0

    return -total_r + penalty + inactive_penalty

def _optimize_grid(nx, ny, w, h, time_budget):
    """Optimize grid lines for given aspect ratio and grid dimensions."""
    n_cells = nx * ny
    if n_cells < N:
        return None
        
    # Create active mask
    active_mask = np.zeros(n_cells, dtype=bool)
    if n_cells == N:
        active_mask[:] = True
    else:
        # Mark first N cells as active
        active_mask[:N] = True
        
    # Initial guess: uniform grid
    gx0 = np.linspace(w/nx, w - w/nx, nx - 1)
    gy0 = np.linspace(h/ny, h - h/ny, ny - 1)
    
    z0 = np.concatenate([gx0, gy0])
    
    bounds = []
    for _ in range(nx - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(ny - 1):
        bounds.append((1e-4, h - 1e-4))
        
    def obj(z):
        return _objective_grid(z, w, h, nx, ny, active_mask)
        
    maxiter = int(min(300, time_budget * 20))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
        
    gx_opt = z_opt[0:nx-1]
    gy_opt = z_opt[nx-1:nx-1+ny-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    centers, radii = _cell_params(grid_x, grid_y)
    
    active_idx = np.where(active_mask)[0]
    x = centers[active_idx, 0]
    y = centers[active_idx, 1]
    r = radii[active_idx]
    
    return x, y, r

def _check_validity(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False, 0.0
    return True, float(np.sum(r))

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    # Grid configurations: (nx, ny)
    # 7x3 and 3x7 are exact fits (21 cells)
    # 6x4 and 4x6 have 24 cells (3 inactive)
    # 5x5 has 25 cells (4 inactive)
    grid_configs = [(7, 3), (3, 7), (6, 4), (4, 6), (5, 5)]
    
    # Aspect ratios to try
    # For 7x3, aspect ~ 2.33 is ideal. w+h=2.
    # w/7 ~ h/3 => 3w ~ 7h => h ~ 0.43w => 1.43w=2 => w~1.4
    # For 3x7, aspect ~ 0.43. w~0.6
    # For 5x5, aspect ~ 1.0. w~1.0
    # We will try a range of aspect ratios for each config, or just the "ideal" one plus neighbors.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 1.4, 0.6, 2.0, 0.5]
    
    for nx, ny in grid_configs:
        for w in aspect_ratios:
            if time.time() > deadline - 10:
                break
            h = 2.0 - w
            if h < 0.3:
                continue
            
            remaining = deadline - (time.time() - start_time)
            if remaining < 2:
                break
                
            try:
                res = _optimize_grid(nx, ny, w, h, remaining)
                if res is None:
                    continue
                x, y, r = res
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
            except Exception:
                pass
        
        if time.time() > deadline - 10:
            break

    if best_result is None:
        # Fallback to simple 5x5 grid
        w, h = 1.0, 1.0
        x = np.array([(i % 5) * 0.2 + 0.1 for i in range(N)])
        y = np.array([(i // 5) * 0.2 + 0.1 for i in range(N)])
        r = np.full(N, 0.09)
        valid, val = _check_validity(x, y, r, w, h)
        if valid:
            best_result = (x.copy(), y.copy(), r.copy(), w)
    
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
    
    x, y, r, w = best_result
    h = 2.0 - w
    
    # Final slight shrink to ensure validity
    for _ in range(5):
        rt = r * (1 - 1e-8)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
