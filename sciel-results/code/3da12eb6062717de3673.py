"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with movable circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY

def _compute_cell_bounds(grid_x, grid_y):
    """Compute left, right, bottom, top for each cell."""
    left = np.zeros(N_CELLS)
    right = np.zeros(N_CELLS)
    bottom = np.zeros(N_CELLS)
    top = np.zeros(N_CELLS)
    for r in range(NY):
        for c in range(NX):
            idx = r * NX + c
            left[idx] = grid_x[c]
            right[idx] = grid_x[c + 1]
            bottom[idx] = grid_y[r]
            top[idx] = grid_y[r + 1]
    return left, right, bottom, top

def _objective(z, w, h, active_mask):
    """
    z layout:
      [gx_0..gx_{NX-2}, gy_0..gy_{NY-2}]  : grid lines
      [dx_0..dx_{N-1}, dy_0..dy_{N-1}]    : offsets of each circle within its cell
      [r_0..r_{N-1}]                      : radii
    """
    n_gx = NX - 1
    n_gy = NY - 1
    n_grid = n_gx + n_gy
    
    gx = z[0:n_gx]
    gy = z[n_gx:n_gx + n_gy]
    
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e6
    
    left, right, bottom, top = _compute_cell_bounds(grid_x, grid_y)
    
    # Map each circle to its cell
    cell_idx = np.arange(N)
    
    dx = z[n_grid:n_grid + N]
    dy = z[n_grid + N:n_grid + 2 * N]
    r = z[n_grid + 2 * N:n_grid + 3 * N]
    
    # Circle centers
    cx = left[cell_idx] + (right[cell_idx] - left[cell_idx]) / 2.0 + dx
    cy = bottom[cell_idx] + (top[cell_idx] - bottom[cell_idx]) / 2.0 + dy
    
    # Max allowed radius for each circle given its position within cell
    max_r_x = np.minimum(cx - left[cell_idx], right[cell_idx] - cx)
    max_r_y = np.minimum(cy - bottom[cell_idx], top[cell_idx] - cy)
    max_r = np.minimum(max_r_x, max_r_y)
    
    # Penalty if radius exceeds what fits in cell
    r_exceed = np.maximum(r - max_r, 0.0)
    cell_penalty = 100.0 * np.sum(r_exceed * r_exceed)
    
    # Only consider active circles
    active = active_mask
    cx_a = cx[active]
    cy_a = cy[active]
    r_a = r[active]
    
    total_r = np.sum(r_a)
    
    # Overlap penalty between active circles
    n_a = np.sum(active)
    if n_a < 2:
        return -total_r + cell_penalty
    
    dx_a = cx_a[:, None] - cx_a[None, :]
    dy_a = cy_a[:, None] - cy_a[None, :]
    d_sq = dx_a * dx_a + dy_a * dy_a
    r_sum = r_a[:, None] + r_a[None, :]
    
    iu = np.triu_indices(int(n_a), k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    overlap_penalty = 1000.0 * np.sum(overlap * overlap)
    
    # Penalty for inactive circles having non-zero radius
    inactive = ~active
    r_inactive = r[inactive]
    inactive_penalty = 50.0 * np.sum(r_inactive * r_inactive)
    
    return -total_r + cell_penalty + overlap_penalty + inactive_penalty

def _optimize_grid(w, h, active_mask, time_budget, seed=0):
    """Optimize grid, offsets, and radii for given aspect ratio and active cell selection."""
    rng = np.random.RandomState(seed)
    
    # Initial guess
    gx0 = np.linspace(w / NX, w - w / NX, NX - 1)
    gy0 = np.linspace(h / NY, h - h / NY, NY - 1)
    dx0 = np.zeros(N)
    dy0 = np.zeros(N)
    r0 = np.zeros(N)
    for i in range(N):
        if active_mask[i]:
            r0[i] = min(w / NX, h / NY) / 2.0 * 0.9
    
    z0 = np.concatenate([gx0, gy0, dx0, dy0, r0])
    
    n_gx = NX - 1
    n_gy = NY - 1
    
    bounds = []
    for _ in range(n_gx):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(n_gy):
        bounds.append((1e-4, h - 1e-4))
    for _ in range(2 * N):
        bounds.append((-w, w))
    for _ in range(N):
        bounds.append((0.0, min(w, h) / 2.0))
    
    def obj(z):
        return _objective(z, w, h, active_mask)
    
    maxiter = int(min(200, time_budget * 15))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
    
    n_grid = n_gx + n_gy
    gx_opt = z_opt[0:n_gx]
    gy_opt = z_opt[n_gx:n_gx + n_gy]
    dx_opt = z_opt[n_grid:n_grid + N]
    dy_opt = z_opt[n_grid + N:n_grid + 2 * N]
    r_opt = z_opt[n_grid + 2 * N:n_grid + 3 * N]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    left, right, bottom, top = _compute_cell_bounds(grid_x, grid_y)
    cell_idx = np.arange(N)
    
    cx = left[cell_idx] + (right[cell_idx] - left[cell_idx]) / 2.0 + dx_opt
    cy = bottom[cell_idx] + (top[cell_idx] - bottom[cell_idx]) / 2.0 + dy_opt
    
    # Clip radii to fit in cells
    max_r_x = np.minimum(cx - left[cell_idx], right[cell_idx] - cx)
    max_r_y = np.minimum(cy - bottom[cell_idx], top[cell_idx] - cy)
    r_opt = np.minimum(r_opt, np.minimum(max_r_x, max_r_y))
    r_opt = np.maximum(r_opt, 0.0)
    
    # Only return active circles
    active = active_mask
    x = cx[active]
    y = cy[active]
    r = r_opt[active]
    
    return x, y, r

def _check_validity(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    for i in range(len(x)):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False, 0.0
    return True, float(np.sum(r))

def _generate_active_masks():
    """Generate several patterns of which 21 cells out of 25 are active."""
    masks = []
    
    # All except 4 corners
    mask = np.ones(N_CELLS, dtype=bool)
    for corner in [0, NX - 1, (NY - 1) * NX, (NY - 1) * NX + NX - 1]:
        mask[corner] = False
    masks.append(mask.copy())
    
    # All except 4 cells in one row
    mask = np.ones(N_CELLS, dtype=bool)
    for c in range(NX):
        mask[0 * NX + c] = False
        if np.sum(mask) == N:
            break
    masks.append(mask.copy())
    
    # All except 4 random cells
    rng = np.random.RandomState(42)
    for trial in range(20):
        mask = np.ones(N_CELLS, dtype=bool)
        inactive = rng.choice(N_CELLS, size=N_CELLS - N, replace=False)
        mask[inactive] = False
        if np.sum(mask) == N:
            masks.append(mask.copy())
    
    # Unique masks
    unique_masks = []
    for m in masks:
        key = tuple(m)
        if key not in [tuple(u) for u in unique_masks]:
            unique_masks.append(m)
    
    return unique_masks

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.7, 1.3]
    
    active_masks = _generate_active_masks()
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        for mask in active_masks[:5]:
            if time.time() > deadline - 10:
                break
            
            remaining = deadline - (time.time() - start_time)
            if remaining < 3:
                break
            
            try:
                x, y, r = _optimize_grid(w, h, mask, remaining, seed=int(w * 1000))
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: uniform 5x5 grid, drop 4 corners
        w, h = 1.0, 1.0
        mask = np.ones(N_CELLS, dtype=bool)
        for corner in [0, NX - 1, (NY - 1) * NX, (NY - 1) * NX + NX - 1]:
            mask[corner] = False
        active_idx = np.where(mask)[0]
        
        cell_w = w / NX
        cell_h = h / NY
        r_val = min(cell_w, cell_h) / 2.0 * 0.95
        
        x = np.array([(c * cell_w + cell_w / 2.0) for r in range(NY) for c in range(NX) if mask[r * NX + c]])
        y = np.array([(r * cell_h + cell_h / 2.0) for r in range(NY) for c in range(NX) if mask[r * NX + c]])
        r = np.full(len(x), r_val)
        
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
