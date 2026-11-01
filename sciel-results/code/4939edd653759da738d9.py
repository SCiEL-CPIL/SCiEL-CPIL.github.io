import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE_MARGIN = 10.0 # Seconds to spare before 120s limit

def _is_valid(cx, cy, r, W, H, tol=1e-9):
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _force_relax(x, y, r, w, h, max_iter=2000, dt_init=0.005, time_limit=None, start_time=None):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    """
    n = N
    dt = dt_init
    iu = np.triu_indices(n, k=1)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Boundary forces
        # Push circles away from walls if they are too close (violating r)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        # Strong penalty for boundary violations
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100
        
        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        # Avoid division by zero
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) < 1e-9:
            # Check if boundaries are also satisfied
            if np.max(left_viol) < 1e-9 and np.max(right_viol) < 1e-9 and \
               np.max(bottom_viol) < 1e-9 and np.max(top_viol) < 1e-9:
                break
        
        # Force is proportional to overlap
        # Using a linear force is standard, but a quadratic or stronger force can help escape local minima
        f_pairs = overlap * 20.0 
        
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        
        # Unit vectors
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        
        # Apply forces
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]
            
        # Adaptive step size
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 1.0:
            dt = max(0.001, dt * 0.8)
        elif max_force < 0.1:
            dt = min(0.01, dt * 1.05)
            
        x = x + dt * fx
        y = y + dt * fy
        
        # Clip to boundaries to prevent flying out
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
    return x, y

def _grow_and_relax(x, y, r, w, h, time_limit, start_time):
    """
    Iteratively grow radii and relax positions.
    """
    current_sum = np.sum(r)
    grow_factor = 1.002
    min_grow_factor = 1.0001
    
    while time.time() < time_limit - 5.0:
        # Try to grow radii
        new_r = r * grow_factor
        # Check if we can relax back to a valid state
        new_x, new_y = _force_relax(x.copy(), y.copy(), new_r, w, h, 
                                     max_iter=500, dt_init=0.005, 
                                     time_limit=time_limit - 2.0, 
                                     start_time=start_time)
        
        if _is_valid(new_x, new_y, new_r, w, h):
            x, y, r = new_x, new_y, new_r
            current_sum = np.sum(r)
            # If we succeeded, try to grow a bit more next time? 
            # Or keep the same factor? Let's keep it constant for stability.
        else:
            # Failed to relax, reduce growth factor
            grow_factor *= 0.9
            if grow_factor < min_grow_factor:
                break
            # Also, try to relax the current state a bit more to see if it helps
            x, y = _force_relax(x, y, r, w, h, max_iter=200, dt_init=0.002,
                                time_limit=time_limit - 2.0, start_time=start_time)
            
    return x, y, r

def _objective(z, w, h, penalty_weight=10000.0):
    """
    Objective for SLSQP: minimize -sum(r) + penalties
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    
    if np.any(r <= 0):
        return 1e8
        
    obj = -np.sum(r)
    
    # Overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)
    
    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))
                             
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin for safety
    
    best_value = 0.0
    best_result = None
    
    # Aspect ratios to try
    # W + H = 2. W ranges from 0.5 to 1.5 (symmetric around 1.0)
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 20.0:
            break
        h0 = 2.0 - w0
        
        # Try different grid configurations
        # We need to place 21 circles.
        # Common grids: 4x6 (24 cells), 3x7 (21 cells), 5x5 (25 cells)
        # We will try to fit 21 circles into a grid of size >= 21
        
        configurations = [
            (3, 7), # 21 cells
            (4, 6), # 24 cells (remove 3)
            (5, 5), # 25 cells (remove 4)
            (6, 4), # 24 cells
            (7, 3), # 21 cells
            (4, 5), # 20 cells (need to add 1? No, must be disjoint. If grid is 20, we can't place 21. 
                    # So we must use grids with at least 21 cells.
                    # Actually, we can use a grid with MORE cells and just pick 21 of them?
                    # No, the circles must be disjoint. If we place them in grid cells, they are disjoint if r < min(cell_w, cell_h)/2.
                    # So we need a grid with at least 21 cells.
                    # 3x7=21, 4x6=24, 5x5=25, 6x4=24, 7x3=21, 4x7=28, 5x6=30...
        ]
        
        configurations = [
            (3, 7), (7, 3),
            (4, 6), (6, 4),
            (5, 5),
            (4, 7), (7, 4),
            (5, 6), (6, 5)
        ]
        
        for rows, cols in configurations:
            if time.time() > deadline - 5.0:
                break
                
            if rows * cols < N:
                continue
                
            cell_w = w0 / cols
            cell_h = h0 / rows
            
            # Initial radius estimate
            r_init = min(cell_w, cell_h) * 0.45
            
            # Initialize positions
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            # Place circles in the first N cells of the grid
            # To be more symmetric, we might want to center the 21 circles in the grid.
            # For simplicity, we just fill row by row.
            for i in range(N):
                row = i // cols
                col = i % cols
                x[i] = (col + 0.5) * cell_w
                y[i] = (row + 0.5) * cell_h
                r[i] = r_init
            
            # Relax
            x, y = _force_relax(x, y, r, w0, h0, max_iter=500, dt_init=0.005,
                                time_limit=deadline - 10.0, start_time=start_time)
            
            # Check validity
            if not _is_valid(x, y, r, w0, h0):
                # Try to shrink to make valid
                scale = 0.9
                while not _is_valid(x, y, r * scale, w0, h0) and scale > 0.1:
                    scale -= 0.05
                if scale <= 0.1:
                    continue
                r = r * scale
            
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)
            
            # Grow and Relax
            x, y, r = _grow_and_relax(x, y, r, w0, h0, deadline - 15.0, start_time)
            
            if _is_valid(x, y, r, w0, h0):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w0, h0)
            
            # SLSQP Polish
            if time.time() < deadline - 10.0:
                z = np.zeros(3 * N)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                
                bounds = [(0.01, w0 - 0.01)] * N + [(0.01, h0 - 0.01)] * N + [(0.01, 1.0)] * N
                
                try:
                    res = minimize(_objective, z, args=(w0, h0), method='SLSQP',
                                   bounds=bounds,
                                   options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 50})
                    z_opt = res.x
                    x_opt = z_opt[0::3][:N]
                    y_opt = z_opt[1::3][:N]
                    r_opt = z_opt[2::3][:N]
                    
                    if _is_valid(x_opt, y_opt, r_opt, w0, h0):
                        val = float(np.sum(r_opt))
                        if val > best_value:
                            best_value = val
                            best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w0, h0)
                    else:
                        # Try to repair
                        for scale in [0.98, 0.95, 0.9]:
                            r_rep = r_opt * scale
                            if _is_valid(x_opt, y_opt, r_rep, w0, h0):
                                val = float(np.sum(r_rep))
                                if val > best_value:
                                    best_value = val
                                    best_result = (x_opt.copy(), y_opt.copy(), r_rep.copy(), w0, h0)
                                break
                except Exception:
                    pass
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
