"""Circle packing in perimeter-4 rectangle: Optimized Grid Cellular Decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _softmax_widths(u, total_w):
    """Convert unconstrained u to positive widths summing to total_w."""
    e_u = np.exp(u - np.max(u))
    return e_u / np.sum(e_u) * total_w

def _objective_grid_dims(z, W, H, NX, NY):
    """
    Objective function for optimizing grid dimensions.
    z: [u_0...u_{NX-1}, v_0...v_{NY-1}]
    """
    u = z[:NX]
    v = z[NX:NX+NY]
    
    w = _softmax_widths(u, W)
    h = _softmax_widths(v, H)
    
    # Compute inradii matrix: M[i,j] = min(w[i], h[j]) / 2
    # w is (NX,), h is (NY,)
    # M is (NX, NY)
    M = np.minimum(w[:, None], h[None, :]) / 2.0
    
    # Get top 21 inradii
    flat_M = M.flatten()
    # np.partition is faster than sort for top-k
    # We want the largest 21, so we partition at len-21
    if len(flat_M) >= 21:
        # kth is the index of the 21st largest element
        # np.partition returns array where elements at indices < k are <= kth element
        # We want the largest 21, so we look at the last 21 of the partitioned array?
        # Actually, np.partition(a, k) puts the k-th smallest element at index k.
        # So to get the 21 largest, we can partition at len-21 and take the last 21?
        # Or just sort, N is small (e.g., 30-50 cells).
        top_r = np.sort(flat_M)[-21:]
        sum_r = np.sum(top_r)
    else:
        sum_r = np.sum(flat_M)
        
    return -sum_r

def _optimize_grid(W, H, NX, NY, time_budget):
    """
    Optimize grid lines for given rectangle dimensions and grid size.
    """
    # Initial guess: uniform grid
    u0 = np.zeros(NX)
    v0 = np.zeros(NY)
    z0 = np.concatenate([u0, v0])
    
    def obj(z):
        return _objective_grid_dims(z, W, H, NX, NY)
        
    # No bounds needed due to softmax parameterization
    maxiter = int(min(500, time_budget * 50))
    
    try:
        res = minimize(obj, z0, method='Nelder-Mead', 
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
        
    # Reconstruct widths and heights
    u = z_opt[:NX]
    v = z_opt[NX:NX+NY]
    
    w = _softmax_widths(u, W)
    h = _softmax_widths(v, H)
    
    # Compute inradii matrix
    M = np.minimum(w[:, None], h[None, :]) / 2.0
    
    # Find top 21 cells
    flat_M = M.flatten()
    # Get indices of top 21
    # np.argsort returns indices that would sort the array
    idx_sorted = np.argsort(flat_M)[::-1]
    top_idx = idx_sorted[:21]
    
    # Map flat indices to (i, j)
    centers = []
    radii = []
    
    # Compute cumulative sums for centers
    x_cum = np.concatenate(([0.0], np.cumsum(w)))
    y_cum = np.concatenate(([0.0], np.cumsum(h)))
    
    for idx in top_idx:
        i = idx // NY
        j = idx % NY
        
        cx = (x_cum[i] + x_cum[i+1]) / 2.0
        cy = (y_cum[j] + y_cum[j+1]) / 2.0
        r = M[i, j]
        
        centers.append([cx, cy])
        radii.append(r)
        
    x = np.array([c[0] for c in centers])
    y = np.array([c[1] for c in centers])
    r = np.array(radii)
    
    return x, y, r, W, H

def _check_validity(x, y, r, W, H, tol=1e-9):
    """Check if the packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False, 0.0
    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
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
    
    # Grid shapes (NX, NY) to try
    # We need NX*NY >= 21
    # Aspect ratio of grid should roughly match aspect ratio of rectangle
    grid_shapes = [
        (4, 6), (6, 4),
        (5, 5),
        (3, 7), (7, 3),
        (4, 7), (7, 4),
        (5, 6), (6, 5),
        (3, 8), (8, 3),
        (4, 8), (8, 4),
        (5, 7), (7, 5),
        (6, 6)
    ]
    
    # Aspect ratios W (H = 2 - W)
    # Perimeter 4 => W+H=2
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.9, 1.1, 0.75, 1.25]
    
    for W in aspect_ratios:
        if time.time() > deadline - 5:
            break
        H = 2.0 - W
        if H < 0.3:
            continue
            
        for NX, NY in grid_shapes:
            if time.time() > deadline - 2:
                break
                
            # Skip shapes that are very ill-suited for the aspect ratio
            # e.g. if W/H is 2, a 3x7 grid (tall) might be bad if W is wide
            # But let's just try them all, the optimizer will handle it
            # However, to save time, we can filter
            rect_aspect = W / H
            grid_aspect = NX / NY
            # Allow some flexibility
            if rect_aspect > 1.5 and grid_aspect < 0.7:
                continue
            if rect_aspect < 0.67 and grid_aspect > 1.3:
                continue
                
            remaining = deadline - (time.time() - start_time)
            if remaining < 1:
                break
                
            try:
                x, y, r, W_opt, H_opt = _optimize_grid(W, H, NX, NY, remaining)
                
                valid, val = _check_validity(x, y, r, W_opt, H_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), W_opt)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback
        W, H = 1.0, 1.0
        x = np.array([(i % 5) * 0.2 + 0.1 for i in range(N)])
        y = np.array([(i // 5) * 0.2 + 0.1 for i in range(N)])
        r = np.full(N, 0.09)
        valid, val = _check_validity(x, y, r, W, H)
        if valid:
            best_result = (x.copy(), y.copy(), r.copy(), W)
            
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
        
    x, y, r, W = best_result
    H = 2.0 - W
    
    # Final slight shrink to ensure validity
    for _ in range(5):
        rt = r * (1 - 1e-8)
        if _check_validity(x, y, rt, W, H)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
