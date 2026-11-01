"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY
ACTIVE_CELLS = list(range(N))
INACTIVE_CELLS = list(range(N, N_CELLS))

def _compute_cell_params(grid_x, grid_y):
    """Compute centers and inradii for all cells in the grid."""
    n_cells = len(grid_x) - 1
    m_cells = len(grid_y) - 1
    total = n_cells * m_cells
    
    centers = np.zeros((total, 2))
    radii = np.zeros(total)
    
    idx = 0
    for r in range(m_cells):
        for c in range(n_cells):
            x_left = grid_x[c]
            x_right = grid_x[c + 1]
            y_bottom = grid_y[r]
            y_top = grid_y[r + 1]
            
            cx = (x_left + x_right) / 2.0
            cy = (y_bottom + y_top) / 2.0
            w_cell = x_right - x_left
            h_cell = y_top - y_bottom
            r_cell = min(w_cell, h_cell) / 2.0
            
            centers[idx] = [cx, cy]
            radii[idx] = r_cell
            idx += 1
            
    return centers, radii

def _objective(z, w, h, active_mask):
    """
    Objective: maximize sum of active cell inradii.
    z: grid line positions [gx_0..gx_{NX-2}, gy_0..gy_{NY-2}]
    """
    n_gx = NX - 1
    n_gy = NY - 1
    gx = z[0:n_gx]
    gy = z[n_gx:n_gx + n_gy]
    
    # Check for valid sorted grid lines with minimum spacing
    if np.any(np.diff(gx) < 1e-5) or np.any(np.diff(gy) < 1e-5):
        return 1e6
    if gx[0] < 1e-5 or gx[-1] > w - 1e-5:
        return 1e6
    if gy[0] < 1e-5 or gy[-1] > h - 1e-5:
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    # Check minimum cell size
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e6
        
    centers, radii = _compute_cell_params(grid_x, grid_y)
    
    # Sum of active radii
    total_r = np.sum(radii[active_mask])
    
    # Overlap penalty between active circles
    active_idx = np.where(active_mask)[0]
    active_centers = centers[active_idx]
    active_radii = radii[active_idx]
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
        penalty = 500.0 * np.sum(overlap * overlap)
    else:
        penalty = 0.0
    
    # Penalize inactive cells to encourage them to be small
    inactive_idx = np.where(~active_mask)[0]
    if len(inactive_idx) > 0:
        inactive_penalty = 5.0 * np.sum(radii[inactive_idx])
    else:
        inactive_penalty = 0.0
    
    return -total_r + penalty + inactive_penalty

def _optimize_aspect(w, h, time_budget):
    """Optimize grid lines for a given rectangle aspect ratio."""
    # Create active mask: first N cells are active
    active_mask = np.zeros(N_CELLS, dtype=bool)
    active_mask[:N] = True
    
    # Initial guess: uniform grid
    gx0 = np.linspace(w / NX, w - w / NX, NX - 1)
    gy0 = np.linspace(h / NY, h - h / NY, NY - 1)
    z0 = np.concatenate([gx0, gy0])
    
    # Bounds for grid lines
    bounds = []
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
    
    def obj(z):
        return _objective(z, w, h, active_mask)
    
    # Try multiple optimization strategies
    best_z = z0.copy()
    best_obj = obj(z0)
    
    # Strategy 1: SLSQP from uniform start
    try:
        maxiter = int(min(500, max(50, time_budget * 30)))
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 200})
        if res.fun < best_obj:
            best_z = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    # Strategy 2: Try different initial perturbations
    rng = np.random.RandomState(42)
    for trial in range(3):
        if time_budget < 5:
            break
        # Perturb the uniform grid
        noise_gx = rng.uniform(-0.1, 0.1, NX - 1) * w / NX
        noise_gy = rng.uniform(-0.1, 0.1, NY - 1) * h / NY
        gx_pert = np.clip(gx0 + noise_gx, 1e-3, w - 1e-3)
        gy_pert = np.clip(gy0 + noise_gy, 1e-3, h - 1e-3)
        # Ensure sorted
        gx_pert = np.sort(gx_pert)
        gy_pert = np.sort(gy_pert)
        z_pert = np.concatenate([gx_pert, gy_pert])
        
        try:
            maxiter = int(min(200, max(20, time_budget * 10)))
            res = minimize(obj, z_pert, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 100})
            if res.fun < best_obj:
                best_z = res.x
                best_obj = res.fun
        except Exception:
            pass
    
    # Extract optimal grid lines
    gx_opt = best_z[0:NX-1]
    gy_opt = best_z[NX-1:NX-1+NY-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    centers, radii = _compute_cell_params(grid_x, grid_y)
    
    active_idx = np.where(active_mask)[0]
    x = centers[active_idx, 0]
    y = centers[active_idx, 1]
    r = radii[active_idx]
    
    return x, y, r

def _check_validity(x, y, r, w, h, tol=1e-9):
    """Verify packing validity: circles inside rectangle, no overlaps."""
    if w <= 0.01 or h <= 0.01:
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
    deadline = start_time + 108.0
    
    best_value = 0.0
    best_result = None
    
    # Aspect ratios to try (w + h = 4, so h = 4 - w... wait, perimeter = 2(w+h) = 4, so w+h=2)
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.95, 1.05]
    
    for w in aspect_ratios:
        if time.time() > deadline - 8:
            break
        h = 2.0 - w
        if h < 0.2:
            continue
        
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 3:
            break
        
        try:
            x, y, r = _optimize_aspect(w, h, remaining)
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: simple 5x5 grid in square
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
    
    # Slight shrink to ensure numerical validity
    for scale in [0.9999999, 0.999999, 0.99999]:
        rt = r * scale
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
            break
    
    return np.column_stack((x, y, r))
