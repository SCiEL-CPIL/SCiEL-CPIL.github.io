"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _compute_grid_layout(grid_x, grid_y, nx, ny, active_mask):
    """
    Compute circle centers and radii from grid layout.
    active_mask: boolean array of shape (ny, nx) indicating which cells are active.
    Returns centers (N, 2), radii (N,) for active cells only.
    """
    centers_list = []
    radii_list = []
    for r in range(ny):
        for c in range(nx):
            if active_mask[r, c]:
                x_left = grid_x[c]
                x_right = grid_x[c + 1]
                y_bottom = grid_y[r]
                y_top = grid_y[r + 1]
                cx = (x_left + x_right) / 2.0
                cy = (y_bottom + y_top) / 2.0
                r_cell = min(x_right - x_left, y_top - y_bottom) / 2.0
                centers_list.append([cx, cy])
                radii_list.append(r_cell)
    centers = np.array(centers_list)
    radii = np.array(radii_list)
    return centers, radii

def _grid_objective(z, w, h, nx, ny, active_mask):
    """
    Objective: maximize sum of inradii of active cells.
    z: [gx_0..gx_{nx-2}, gy_0..gy_{ny-2}]
    """
    n_gx = nx - 1
    n_gy = ny - 1
    gx = z[0:n_gx]
    gy = z[n_gx:n_gx + n_gy]
    
    # Check sorted and minimum spacing
    if np.any(np.diff(gx) < 1e-5) or np.any(np.diff(gy) < 1e-5):
        return 1e8
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e8
    
    centers, radii = _compute_grid_layout(grid_x, grid_y, nx, ny, active_mask)
    
    if len(radii) != N:
        return 1e8
    
    total_r = np.sum(radii)
    
    # Circles in adjacent cells can overlap if the cell inradius is larger than half the distance
    # Actually, in a grid, circles in different cells are separated by at least the grid line.
    # The distance between centers of adjacent cells is (w_cell1/2 + w_cell2/2) in one direction
    # and h_cell/2 in the other. The sum of radii is (min(w1,h)/2 + min(w2,h)/2).
    # These can overlap! So we need to check.
    
    # Check for overlaps between all pairs
    if N > 1:
        dx = centers[:, 0, None] - centers[None, :, 0]
        dy = centers[:, 1, None] - centers[None, :, 1]
        d_sq = dx * dx + dy * dy
        r_sum = radii[:, None] + radii[None, :]
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        r_sum_pairs = r_sum[iu]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        penalty = 10000.0 * np.sum(overlap * overlap)
    else:
        penalty = 0.0
    
    return -total_r + penalty

def _optimize_grid_config(w, h, nx, ny, active_mask, time_budget):
    """Optimize grid lines for a given configuration."""
    n_gx = nx - 1
    n_gy = ny - 1
    
    # Uniform initial guess
    gx0 = np.linspace(w / nx, w - w / nx, n_gx)
    gy0 = np.linspace(h / ny, h - h / ny, n_gy)
    z0 = np.concatenate([gx0, gy0])
    
    bounds = [(1e-4, w - 1e-4)] * n_gx + [(1e-4, h - 1e-4)] * n_gy
    
    def obj(z):
        return _grid_objective(z, w, h, nx, ny, active_mask)
    
    maxiter = int(min(500, time_budget * 30))
    
    best_z = z0.copy()
    best_val = obj(z0)
    
    # Try multiple starting points
    start_points = [z0]
    # Perturbed starts
    for _ in range(3):
        pert = z0 + 0.1 * np.random.randn(len(z0))
        pert = np.clip(pert, 1e-4, max(w, h) - 1e-4)
        start_points.append(pert)
    
    for sp in start_points:
        if time.time() > time_budget:
            break
        try:
            res = minimize(obj, sp, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
            val = res.fun
            if val < best_val:
                best_val = val
                best_z = res.x.copy()
        except Exception:
            pass
    
    gx_opt = best_z[0:n_gx]
    gy_opt = best_z[n_gx:n_gx + n_gy]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    centers, radii = _compute_grid_layout(grid_x, grid_y, nx, ny, active_mask)
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
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
    
    # Define grid configurations: (nx, ny, active_mask)
    # active_mask is a (ny, nx) boolean array
    
    configs = []
    
    # 7x3 = 21, all active
    mask_73 = np.ones((3, 7), dtype=bool)
    configs.append((7, 3, mask_73))
    
    # 3x7 = 21, all active
    mask_37 = np.ones((7, 3), dtype=bool)
    configs.append((3, 7, mask_37))
    
    # 5x5 = 25, deactivate 4 cells (corners)
    mask_55 = np.ones((5, 5), dtype=bool)
    mask_55[0, 0] = False
    mask_55[0, 4] = False
    mask_55[4, 0] = False
    mask_55[4, 4] = False
    configs.append((5, 5, mask_55))
    
    # 5x5 = 25, deactivate 4 cells (middle of edges)
    mask_55b = np.ones((5, 5), dtype=bool)
    mask_55b[0, 2] = False
    mask_55b[4, 2] = False
    mask_55b[2, 0] = False
    mask_55b[2, 4] = False
    configs.append((5, 5, mask_55b))
    
    # 6x4 = 24, deactivate 3 cells
    mask_64 = np.ones((4, 6), dtype=bool)
    mask_64[0, 0] = False
    mask_64[0, 5] = False
    mask_64[3, 0] = False
    configs.append((6, 4, mask_64))
    
    # 4x6 = 24, deactivate 3 cells
    mask_46 = np.ones((6, 4), dtype=bool)
    mask_46[0, 0] = False
    mask_46[0, 3] = False
    mask_46[5, 0] = False
    configs.append((4, 6, mask_46))
    
    # 9x3 = 27, deactivate 6 cells
    mask_93 = np.ones((3, 9), dtype=bool)
    mask_93[0, 0] = False
    mask_93[0, 1] = False
    mask_93[0, 7] = False
    mask_93[0, 8] = False
    mask_93[2, 0] = False
    mask_93[2, 8] = False
    configs.append((9, 3, mask_93))
    
    # 3x9 = 27, deactivate 6 cells
    mask_39 = np.ones((9, 3), dtype=bool)
    mask_39[0, 0] = False
    mask_39[1, 0] = False
    mask_39[7, 0] = False
    mask_39[8, 0] = False
    mask_39[0, 2] = False
    mask_39[8, 2] = False
    configs.append((3, 9, mask_39))
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 1.4, 0.71]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.3 or w < 0.3:
            continue
        
        for nx, ny, active_mask in configs:
            if time.time() > deadline - 5:
                break
            
            remaining = deadline - time.time()
            if remaining < 3:
                break
            
            # Allocate time per config based on number of configs
            time_per_config = min(remaining / max(1, len(configs)), 15.0)
            
            try:
                x, y, r = _optimize_grid_config(w, h, nx, ny, active_mask, time_per_config)
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: simple 5x5 grid with 4 inactive corners, square
        w, h = 1.0, 1.0
        mask = np.ones((5, 5), dtype=bool)
        mask[0, 0] = False
        mask[0, 4] = False
        mask[4, 0] = False
        mask[4, 4] = False
        
        grid_x = np.linspace(0, w, 6)
        grid_y = np.linspace(0, h, 6)
        centers, radii = _compute_grid_layout(grid_x, grid_y, 5, 5, mask)
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii
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
