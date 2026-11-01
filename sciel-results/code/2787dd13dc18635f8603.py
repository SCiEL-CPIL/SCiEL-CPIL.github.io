"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
# Number of cells used
N_CELLS = NX * NY
# Indices of active cells
ACTIVE_CELLS = list(range(N))
# Indices of inactive cells (to be shrunk/ignored)
INACTIVE_CELLS = list(range(N, N_CELLS))

def _cell_params(grid_x, grid_y):
    """
    Calculate centers and inradii for all cells.
    Returns:
        centers: (N_CELLS, 2) array of [x, y]
        radii: (N_CELLS,) array of inradii
    """
    centers = np.zeros((N_CELLS, 2))
    radii = np.zeros(N_CELLS)
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
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

def _objective_grid(z, w, h):
    """
    Objective function for grid optimization.
    z layout: [gx_0, ..., gx_{NX-2}, gy_0, ..., gy_{NY-2}]
    """
    gx = z[0:NX-1]
    gy = z[NX-1:NX-1+NY-1]
    
    # Ensure grid lines are within bounds and sorted
    # We handle this via bounds in minimize, but check for safety
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    # Check if any cell is too small (to avoid numerical issues)
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e6
        
    centers, radii = _cell_params(grid_x, grid_y)
    
    # Only consider active cells for the sum
    active_idx = np.array(ACTIVE_CELLS)
    active_centers = centers[active_idx]
    active_radii = radii[active_idx]
    
    total_r = np.sum(active_radii)
    
    # Overlap penalty between active circles
    # N=21, so 210 pairs. Vectorized.
    dx = active_centers[:, 0, None] - active_centers[None, :, 0]
    dy = active_centers[:, 1, None] - active_centers[None, :, 1]
    d_sq = dx * dx + dy * dy
    r_sum = active_radii[:, None] + active_radii[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    penalty = 1000.0 * np.sum(overlap * overlap)
    
    # Penalty for inactive cells having significant size?
    # If inactive cells are large, they might force active cells to be small.
    # We don't explicitly penalize their size, but if they are large, they take space.
    # The optimizer should shrink them to 0 if possible.
    # However, if we don't penalize them, the optimizer might keep them large if it doesn't hurt active cells.
    # But large inactive cells mean the grid lines are far apart, potentially limiting active cells?
    # No, active cells are independent in terms of size, except they share grid lines.
    # If an inactive cell is large, it consumes "width/height" from the rectangle.
    # So we should encourage inactive cells to be small.
    inactive_idx = np.array(INACTIVE_CELLS)
    inactive_radii = radii[inactive_idx]
    # Penalize the sum of inactive radii to shrink them
    inactive_penalty = 10.0 * np.sum(inactive_radii)
    
    return -total_r + penalty + inactive_penalty

def _optimize_grid(w, h, time_budget):
    """Optimize grid lines for given aspect ratio."""
    # Initial guess: uniform grid
    gx0 = np.linspace(w/NX, w - w/NX, NX - 1)
    gy0 = np.linspace(h/NY, h - h/NY, NY - 1)
    
    z0 = np.concatenate([gx0, gy0])
    
    bounds = []
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
        
    def obj(z):
        return _objective_grid(z, w, h)
        
    # Use SLSQP for small dimension problem
    maxiter = int(min(300, time_budget * 20))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
        
    gx_opt = z_opt[0:NX-1]
    gy_opt = z_opt[NX-1:NX-1+NY-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    centers, radii = _cell_params(grid_x, grid_y)
    
    active_idx = np.array(ACTIVE_CELLS)
    x = centers[active_idx, 0]
    y = centers[active_idx, 1]
    r = radii[active_idx]
    
    return x, y, r, grid_x, grid_y

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
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15]
    
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
            x, y, r, grid_x, grid_y = _optimize_grid(w, h, remaining)
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to simple grid
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
