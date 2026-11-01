"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with optimized layouts."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _compute_cell_params(grid_x, grid_y):
    """
    Compute centers and max possible radii for each cell.
    Returns:
        centers: (n_cells, 2) array
        max_radii: (n_cells,) array of inradii
    """
    nx = len(grid_x) - 1
    ny = len(grid_y) - 1
    n_cells = nx * ny
    centers = np.zeros((n_cells, 2))
    max_radii = np.zeros(n_cells)
    
    idx = 0
    for r in range(ny):
        for c in range(nx):
            x0, x1 = grid_x[c], grid_x[c+1]
            y0, y1 = grid_y[r], grid_y[r+1]
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            w_cell = x1 - x0
            h_cell = y1 - y0
            r_cell = min(w_cell, h_cell) / 2.0
            centers[idx] = [cx, cy]
            max_radii[idx] = r_cell
            idx += 1
    
    return centers, max_radii

def _grid_objective(z, w, h, active_mask):
    """
    Objective for grid line optimization.
    z: [gx_0..gx_{nx-2}, gy_0..gy_{ny-2}]
    Returns negative sum of active radii + penalties.
    """
    nx = len(active_mask) // 1  # We'll infer from z length
    # Parse z
    nx_inner = len(z) // 2
    ny_inner = len(z) - nx_inner
    nx = nx_inner + 1
    ny = ny_inner + 1
    
    gx = z[:nx_inner]
    gy = z[nx_inner:]
    
    if np.any(np.diff(gx) < 1e-5) or np.any(np.diff(gy) < 1e-5):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 1e-3) or np.any(np.diff(grid_y) < 1e-3):
        return 1e6
    
    centers, max_radii = _compute_cell_params(grid_x, grid_y)
    
    # active_mask is boolean array of length nx*ny
    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        return 1e6
    
    ac = centers[active_idx]
    ar = max_radii[active_idx]
    
    total_r = np.sum(ar)
    
    # Overlap penalty
    n_act = len(active_idx)
    if n_act > 1:
        dx = ac[:, 0, None] - ac[None, :, 0]
        dy = ac[:, 1, None] - ac[None, :, 1]
        d_sq = dx*dx + dy*dy
        r_sum = ar[:, None] + ar[None, :]
        iu = np.triu_indices(n_act, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        r_sum_pairs = r_sum[iu]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        penalty = 500.0 * np.sum(overlap * overlap)
    else:
        penalty = 0.0
    
    # Penalize inactive cells to shrink them
    inactive_idx = np.where(~active_mask)[0]
    if len(inactive_idx) > 0:
        inactive_r = max_radii[inactive_idx]
        inactive_penalty = 5.0 * np.sum(inactive_r)
    else:
        inactive_penalty = 0.0
    
    return -total_r + penalty + inactive_penalty

def _optimize_for_grid(nx, ny, w, h, time_budget, active_mask=None):
    """
    Optimize grid lines for given nx, ny grid in rectangle w x h.
    If active_mask is None, use first N cells.
    """
    if active_mask is None:
        active_mask = np.zeros(nx * ny, dtype=bool)
        active_mask[:N] = True
    
    nx_inner = nx - 1
    ny_inner = ny - 1
    z_len = nx_inner + ny_inner
    
    # Multiple initial guesses
    best_obj = 1e18
    best_z = None
    
    # Uniform
    gx0 = np.linspace(w/nx, w - w/nx, nx_inner)
    gy0 = np.linspace(h/ny, h - h/ny, ny_inner)
    z0 = np.concatenate([gx0, gy0])
    
    # Slightly perturbed
    rng = np.random.RandomState(42)
    candidates = [z0]
    for _ in range(3):
        gx_p = gx0 + rng.uniform(-0.02*w, 0.02*w, nx_inner)
        gy_p = gy0 + rng.uniform(-0.02*h, 0.02*h, ny_inner)
        candidates.append(np.concatenate([gx_p, gy_p]))
    
    bounds = []
    for _ in range(nx_inner):
        bounds.append((1e-3, w - 1e-3))
    for _ in range(ny_inner):
        bounds.append((1e-3, h - 1e-3))
    
    maxiter = max(50, int(min(200, time_budget * 15)))
    
    for z_init in candidates:
        # Ensure sorted
        if np.any(np.diff(z_init[:nx_inner]) <= 0) or np.any(np.diff(z_init[nx_inner:]) <= 0):
            continue
        if np.any(z_init < 1e-3) or np.any(z_init > max(w, h) - 1e-3):
            continue
        
        try:
            res = minimize(_grid_objective, z_init, args=(w, h, active_mask),
                          method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
            if res.fun < best_obj:
                best_obj = res.fun
                best_z = res.x.copy()
        except Exception:
            pass
    
    if best_z is None:
        # Return uniform grid result
        gx = np.linspace(w/nx, w - w/nx, nx_inner)
        gy = np.linspace(h/ny, h - h/ny, ny_inner)
    else:
        gx = best_z[:nx_inner]
        gy = best_z[nx_inner:]
        # Ensure sorted
        gx = np.sort(gx)
        gy = np.sort(gy)
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    centers, max_radii = _compute_cell_params(grid_x, grid_y)
    
    active_idx = np.where(active_mask)[0]
    x = centers[active_idx, 0]
    y = centers[active_idx, 1]
    r = max_radii[active_idx]
    
    return x, y, r, grid_x, grid_y

def _refine_circles(x, y, r, grid_x, grid_y, active_mask, nx, ny, w, h):
    """
    Refine circle centers within their cells to potentially increase radii.
    For each cell, the circle can be positioned anywhere as long as it fits.
    We optimize center positions to maximize minimum distance to neighbors.
    """
    n_cells = nx * ny
    if len(active_mask) != n_cells:
        return x, y, r
    
    active_idx = np.where(active_mask)[0]
    n_act = len(active_idx)
    
    # For each active circle, determine its cell
    cell_of_circle = active_idx
    
    # Current centers and radii
    cx = x.copy()
    cy = y.copy()
    cr = r.copy()
    
    # For each active circle, compute bounds for center within its cell
    # Cell (row, col): row = cell_idx // nx, col = cell_idx % nx
    bounds_x = []
    bounds_y = []
    for i, cell_idx in enumerate(cell_of_circle):
        row = cell_idx // nx
        col = cell_idx % nx
        x0, x1 = grid_x[col], grid_x[col+1]
        y0, y1 = grid_y[row], grid_y[row+1]
        # Center must be at least r from each wall
        # But r might change... for now keep r fixed and optimize center
        lo_x = max(x0 + cr[i], 0.0)
        hi_x = min(x1 - cr[i], w)
        lo_y = max(y0 + cr[i], 0.0)
        hi_y = min(y1 - cr[i], h)
        if lo_x > hi_x:
            lo_x, hi_x = (x0+x1)/2, (x0+x1)/2
        if lo_y > hi_y:
            lo_y, hi_y = (y0+y1)/2, (y0+y1)/2
        bounds_x.append((lo_x, hi_x))
        bounds_y.append((lo_y, hi_y))
    
    def _refine_obj(z):
        zx = z[:n_act]
        zy = z[n_act:]
        
        # Check boundaries
        for i in range(n_act):
            if zx[i] < bounds_x[i][0] - 1e-9 or zx[i] > bounds_x[i][1] + 1e-9:
                return 1e6
            if zy[i] < bounds_y[i][0] - 1e-9 or zy[i] > bounds_y[i][1] + 1e-9:
                return 1e6
        
        # Overlap penalty
        if n_act > 1:
            dx = zx[:, None] - zx[None, :]
            dy = zy[:, None] - zy[None, :]
            d_sq = dx*dx + dy*dy
            r_sum = cr[:, None] + cr[None, :]
            iu = np.triu_indices(n_act, k=1)
            d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
            r_sum_pairs = r_sum[iu]
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
            return 100.0 * np.sum(overlap * overlap)
        return 0.0
    
    z0 = np.concatenate([cx, cy])
    bounds = []
    for bx, by in zip(bounds_x, bounds_y):
        bounds.append(bx)
        bounds.append(by)
    
    try:
        res = minimize(_refine_obj, z0, method='SLSQP', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-10})
        if res.success or res.fun < _refine_obj(z0):
            cx = res.x[:n_act]
            cy = res.x[n_act:]
    except Exception:
        pass
    
    return cx, cy, cr

def _check_validity(x, y, r, w, h, tol=1e-8):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    if len(x) != N or len(y) != N or len(r) != N:
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
    
    # Try different grid configurations and aspect ratios
    # Grid configs: (nx, ny)
    grid_configs = [
        (5, 5),   # 25 cells, 21 active, 4 inactive
        (4, 6),   # 24 cells, 21 active, 3 inactive
        (6, 4),   # 24 cells
        (3, 7),   # 21 cells exactly
        (7, 3),   # 21 cells
        (4, 5),   # 20 cells - need to handle
    ]
    
    # For configs with fewer than N cells, we need to skip
    # For configs with exactly N cells, all active
    # For configs with more, some inactive
    
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.7, 1.3]
    
    configs_to_try = []
    for nx, ny in grid_configs:
        n_cells = nx * ny
        if n_cells < N:
            continue
        active_mask = np.zeros(n_cells, dtype=bool)
        active_mask[:N] = True
        configs_to_try.append((nx, ny, active_mask))
    
    for w in aspect_ratios:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        if h < 0.25:
            continue
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        for nx, ny, active_mask in configs_to_try:
            if time.time() > deadline - 3:
                break
            rem = deadline - time.time()
            if rem < 2:
                break
            
            try:
                x, y, r, grid_x, grid_y = _optimize_for_grid(nx, ny, w, h, rem, active_mask)
                
                # Refine centers
                x, y, r = _refine_circles(x, y, r, grid_x, grid_y, active_mask, nx, ny, w, h)
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: simple 5x5 grid in square
        w, h = 1.0, 1.0
        # Place 21 circles in 5x5 grid, skip 4 corners
        positions = []
        for i in range(5):
            for j in range(5):
                if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                    continue
                positions.append((j * 0.2 + 0.1, i * 0.2 + 0.1))
        if len(positions) >= N:
            x = np.array([p[0] for p in positions[:N]])
            y = np.array([p[1] for p in positions[:N]])
            r = np.full(N, 0.09)
            valid, val = _check_validity(x, y, r, w, h)
            if valid:
                best_result = (x.copy(), y.copy(), r.copy(), w)
    
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
    
    x, y, r, w = best_result
    h = 2.0 - w
    
    # Final slight shrink to ensure validity
    for _ in range(10):
        rt = r * (1 - 1e-7)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
