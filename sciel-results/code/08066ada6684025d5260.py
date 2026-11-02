"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with optimized cell positions."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY
ACTIVE_CELLS = list(range(N))
INACTIVE_CELLS = list(range(N, N_CELLS))

def _compute_cell_inradii(grid_x, grid_y):
    """Compute inradius for each cell based on grid boundaries."""
    radii = np.zeros(N_CELLS)
    idx = 0
    for r in range(NY):
        for c in range(NX):
            w_cell = grid_x[c + 1] - grid_x[c]
            h_cell = grid_y[r + 1] - grid_y[r]
            radii[idx] = min(w_cell, h_cell) / 2.0
            idx += 1
    return radii

def _get_cell_centers(grid_x, grid_y):
    """Get center positions for each cell."""
    centers = np.zeros((N_CELLS, 2))
    idx = 0
    for r in range(NY):
        for c in range(NX):
            cx = (grid_x[c] + grid_x[c + 1]) / 2.0
            cy = (grid_y[r] + grid_y[r + 1]) / 2.0
            centers[idx] = [cx, cy]
            idx += 1
    return centers

def _objective_full(z, w, h):
    """
    Full objective: optimize grid lines AND circle positions within cells.
    z layout: [gx_0..gx_{NX-2}, gy_0..gy_{NY-2}, dx_0..dx_{N-1}, dy_0..dy_{N-1}]
    where dx, dy are offsets from cell centers for active circles.
    """
    n_gx = NX - 1
    n_gy = NY - 1
    n_pos = 2 * N
    
    gx = z[0:n_gx]
    gy = z[n_gx:n_gx + n_gy]
    dx = z[n_gx + n_gy:n_gx + n_gy + N]
    dy = z[n_gx + n_gy + N:n_gx + n_gy + n_pos]
    
    # Check grid validity
    if np.any(np.diff(gx) < 1e-5) or np.any(np.diff(gy) < 1e-5):
        return 1e8
    if np.any(gx < 1e-5) or np.any(gx > w - 1e-5):
        return 1e8
    if np.any(gy < 1e-5) or np.any(gy > h - 1e-5):
        return 1e8
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    # Check cell sizes
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e8
    
    radii_base = _compute_cell_inradii(grid_x, grid_y)
    centers_base = _get_cell_centers(grid_x, grid_y)
    
    active_idx = np.array(ACTIVE_CELLS)
    base_radii = radii_base[active_idx]
    base_centers = centers_base[active_idx]
    
    # Apply offsets to get actual positions
    x = base_centers[:, 0] + dx
    y = base_centers[:, 1] + dy
    
    # Radii are constrained by cell inradius (can't exceed it)
    r = base_radii.copy()
    
    # Check that circles stay within their cells
    for i in range(N):
        cell_idx = active_idx[i]
        row = cell_idx // NX
        col = cell_idx % NX
        
        # Circle must fit in cell
        cx_min = grid_x[col]
        cx_max = grid_x[col + 1]
        cy_min = grid_y[row]
        cy_max = grid_y[row + 1]
        
        # Check if circle fits horizontally
        if x[i] - r[i] < cx_min - 1e-10 or x[i] + r[i] > cx_max + 1e-10:
            # Shrink radius to fit
            r[i] = min(x[i] - cx_min, cx_max - x[i])
        
        # Check if circle fits vertically
        if y[i] - r[i] < cy_min - 1e-10 or y[i] + r[i] > cy_max + 1e-10:
            # Shrink radius to fit
            r[i] = min(r[i], y[i] - cy_min, cy_max - y[i])
        
        # Final check
        if r[i] < 1e-6:
            return 1e8
    
    total_r = np.sum(r)
    
    # Overlap penalty
    dx_arr = x[:, None] - x[None, :]
    dy_arr = y[:, None] - y[None, :]
    d_sq = dx_arr * dx_arr + dy_arr * dy_arr
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    penalty = 500.0 * np.sum(overlap * overlap)
    
    # Penalize inactive cells (encourage them to be small)
    inactive_idx = np.array(INACTIVE_CELLS)
    inactive_radii = radii_base[inactive_idx]
    inactive_penalty = 5.0 * np.sum(inactive_radii)
    
    return -total_r + penalty + inactive_penalty

def _optimize(w, h, time_budget):
    """Optimize grid and circle positions for given aspect ratio."""
    # Initial guess
    gx0 = np.linspace(w/NX, w - w/NX, NX - 1)
    gy0 = np.linspace(h/NY, h - h/NY, NY - 1)
    dx0 = np.zeros(N)
    dy0 = np.zeros(N)
    
    z0 = np.concatenate([gx0, gy0, dx0, dy0])
    
    bounds = []
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
    # Offsets: allow movement within cell, roughly ±half cell size
    cell_w = w / NX
    cell_h = h / NY
    for _ in range(N):
        bounds.append((-cell_w * 0.4, cell_w * 0.4))
    for _ in range(N):
        bounds.append((-cell_h * 0.4, cell_h * 0.4))
    
    def obj(z):
        return _objective_full(z, w, h)
    
    maxiter = int(min(500, time_budget * 30))
    
    best_z = z0.copy()
    best_f = obj(z0)
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 200})
        if res.fun < best_f:
            best_z = res.x
            best_f = res.fun
    except Exception:
        pass
    
    # Extract results
    n_gx = NX - 1
    n_gy = NY - 1
    gx_opt = best_z[0:n_gx]
    gy_opt = best_z[n_gx:n_gx + n_gy]
    dx_opt = best_z[n_gx + n_gy:n_gx + n_gy + N]
    dy_opt = best_z[n_gx + n_gy + N:n_gx + n_gy + 2*N]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    radii_base = _compute_cell_inradii(grid_x, grid_y)
    centers_base = _get_cell_centers(grid_x, grid_y)
    
    active_idx = np.array(ACTIVE_CELLS)
    base_radii = radii_base[active_idx]
    base_centers = centers_base[active_idx]
    
    x = base_centers[:, 0] + dx_opt
    y = base_centers[:, 1] + dy_opt
    r = base_radii.copy()
    
    # Apply cell constraint
    for i in range(N):
        cell_idx = active_idx[i]
        row = cell_idx // NX
        col = cell_idx % NX
        
        cx_min = grid_x[col]
        cx_max = grid_x[col + 1]
        cy_min = grid_y[row]
        cy_max = grid_y[row + 1]
        
        r[i] = min(r[i], x[i] - cx_min, cx_max - x[i], y[i] - cy_min, cy_max - y[i])
        r[i] = max(r[i], 1e-8)
    
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
    
    # Try various aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 0.95, 1.05]
    
    for w in aspect_ratios:
        if time.time() > deadline - 8:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        try:
            x, y, r = _optimize(w, h, remaining)
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: simple 5x5 grid with 21 circles
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
    for _ in range(10):
        rt = r * (1 - 1e-7)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
    
    return np.column_stack((x, y, r))
