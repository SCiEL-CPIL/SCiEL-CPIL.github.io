"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _compute_radii(w, h, grid_x, grid_y, nx, ny):
    """
    Compute radii for a grid of cells.
    grid_x: array of length nx+1 (0, x1, ..., x_{nx-1}, w)
    grid_y: array of length ny+1 (0, y1, ..., y_{ny-1}, h)
    """
    radii = np.zeros(nx * ny)
    idx = 0
    for i in range(nx):
        w_i = grid_x[i+1] - grid_x[i]
        for j in range(ny):
            h_j = grid_y[j+1] - grid_y[j]
            radii[idx] = 0.5 * min(w_i, h_j)
            idx += 1
    return np.sum(radii)

def _optimize_grid(nx, ny, time_budget):
    """
    Optimize grid lines and aspect ratio for an nx x ny grid.
    Variables: [w, x1, ..., x_{nx-1}, y1, ..., y_{ny-1}]
    h = 2 - w
    """
    n_vars = 1 + (nx - 1) + (ny - 1)
    
    # Initial guess
    w0 = 1.0
    h0 = 1.0
    # Uniform grid
    xs0 = np.linspace(w0 / nx, w0 - w0 / nx, nx - 1)
    ys0 = np.linspace(h0 / ny, h0 - h0 / ny, ny - 1)
    
    z0 = np.concatenate(([w0], xs0, ys0))
    
    # Bounds
    # w in [0.5, 1.5] (since h=2-w must be > 0.5 for symmetry, or just > 0)
    # Actually, w can be anything in (0, 2). But extreme aspect ratios are bad.
    # Let's bound w in [0.4, 1.6]
    bounds = [(0.4, 1.6)]
    for _ in range(nx - 1):
        bounds.append((1e-4, 1.6 - 1e-4)) # x_i < w, but w is variable. 
        # SLSQP bounds must be fixed. We can't have bound depend on w.
        # We must enforce x_i < w via constraints or careful initialization.
        # Better: Use variables for x_i and y_i and w.
        # Constraint: x_i < w.
        # Or, simpler: Optimize x_i, y_i, w.
        # If we use bounds, x_i max is 1.6. But if w=0.5, x_i must be < 0.5.
        # SLSQP can handle inequality constraints.
        
    # Let's use constraints instead of tight bounds for x_i, y_i.
    # Bounds:
    # w: [0.4, 1.6]
    # x_i: [1e-4, 1.6] (will be constrained by w)
    # y_j: [1e-4, 1.6] (will be constrained by h)
    
    for _ in range(nx - 1):
        bounds.append((1e-4, 1.6))
    for _ in range(ny - 1):
        bounds.append((1e-4, 1.6))
        
    def objective(z):
        w = z[0]
        h = 2.0 - w
        if h < 0.4: # Ensure h is not too small
            return 1e6
        xs = z[1:1+nx-1]
        ys = z[1+nx-1:1+nx-1+ny-1]
        
        # Check ordering and bounds
        if np.any(np.diff(xs) < 1e-5) or np.any(np.diff(ys) < 1e-5):
            return 1e6
        if xs[0] < 1e-4 or xs[-1] > w - 1e-4:
            return 1e6
        if ys[0] < 1e-4 or ys[-1] > h - 1e-4:
            return 1e6
            
        grid_x = np.concatenate(([0.0], xs, [w]))
        grid_y = np.concatenate(([0.0], ys, [h]))
        
        # Calculate sum of radii
        total_r = 0.0
        for i in range(nx):
            w_i = grid_x[i+1] - grid_x[i]
            for j in range(ny):
                h_j = grid_y[j+1] - grid_y[j]
                total_r += 0.5 * min(w_i, h_j)
                
        return -total_r

    # Constraints:
    # 1. x_i < w
    # 2. y_j < h = 2 - w
    # These are handled by the objective penalty if we just return 1e6.
    # But SLSQP works better with explicit constraints.
    
    constraints = []
    # x_i < w  =>  x_i - w < 0
    for i in range(nx - 1):
        idx = 1 + i
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, i=idx: z[0] - z[i] - 1e-5
        })
    # y_j < h => y_j - (2 - w) < 0 => w + y_j - 2 < 0
    for j in range(ny - 1):
        idx = 1 + nx - 1 + j
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, j=idx: z[0] + z[j] - 2.0 + 1e-5
        })
        
    # Ordering constraints: x_i < x_{i+1}
    for i in range(nx - 2):
        idx1 = 1 + i
        idx2 = 1 + i + 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, i=idx1, j=idx2: z[j] - z[i] - 1e-5
        })
    for j in range(ny - 2):
        idx1 = 1 + nx - 1 + j
        idx2 = 1 + nx - 1 + j + 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, i=idx1, j=idx2: z[j] - z[i] - 1e-5
        })

    maxiter = int(min(200, time_budget * 10))
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.fun < -0.1: # If we got a reasonable result
            z_opt = res.x
        else:
            z_opt = z0
    except Exception:
        z_opt = z0
        
    w_opt = z_opt[0]
    h_opt = 2.0 - w_opt
    xs_opt = z_opt[1:1+nx-1]
    ys_opt = z_opt[1+nx-1:1+nx-1+ny-1]
    
    grid_x = np.concatenate(([0.0], xs_opt, [w_opt]))
    grid_y = np.concatenate(([0.0], ys_opt, [h_opt]))
    
    # Extract centers and radii
    centers = np.zeros((nx * ny, 2))
    radii = np.zeros(nx * ny)
    idx = 0
    for i in range(nx):
        cx = (grid_x[i] + grid_x[i+1]) / 2.0
        w_i = grid_x[i+1] - grid_x[i]
        for j in range(ny):
            cy = (grid_y[j] + grid_y[j+1]) / 2.0
            h_j = grid_y[j+1] - grid_y[j]
            r = 0.5 * min(w_i, h_j)
            centers[idx] = [cx, cy]
            radii[idx] = r
            idx += 1
            
    return centers, radii, w_opt, h_opt

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    # Try 3x7 and 7x3 grids
    grids = [(3, 7), (7, 3)]
    
    for nx, ny in grids:
        if time.time() > deadline - 5:
            break
        try:
            centers, radii, w, h = _optimize_grid(nx, ny, deadline - time.time())
            val = np.sum(radii)
            if val > best_value:
                best_value = val
                best_result = (centers.copy(), radii.copy(), w, h)
        except Exception:
            pass
            
    if best_result is None:
        # Fallback
        w, h = 1.0, 1.0
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            cx = (i % 7) * (w/7.0) + w/14.0
            cy = (i // 7) * (h/3.0) + h/6.0
            centers[i] = [cx, cy]
            radii[i] = 0.5 * min(w/7.0, h/3.0)
        best_value = np.sum(radii)
        best_result = (centers, radii, w, h)
        
    centers, radii, w, h = best_result
    
    # Final shrink
    for _ in range(10):
        radii_shrunk = radii * (1 - 1e-6)
        # Check validity
        valid = True
        for i in range(N):
            if centers[i][0] < radii_shrunk[i] or centers[i][0] > w - radii_shrunk[i]:
                valid = False
                break
            if centers[i][1] < radii_shrunk[i] or centers[i][1] > h - radii_shrunk[i]:
                valid = False
                break
        if valid:
            radii = radii_shrunk
        else:
            break
            
    result = np.column_stack((centers, radii))
    return result
