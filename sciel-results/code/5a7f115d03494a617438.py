# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_1d_union_length(x, N=8):
    """
    Compute the integral from 0 to 1 of the union length of intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0..N-1.
    
    Uses exact breakpoint analysis.
    """
    # For each c, we have N intervals: [x_i + c*i/N, x_i + 1/N + c*i/N]
    # The union length is a piecewise linear function of c.
    # Breakpoints occur when any left endpoint equals any right endpoint,
    # or any left endpoint equals any left endpoint (which gives same ordering).
    
    # Left_i(c) = x_i + (i/N)*c
    # Right_i(c) = x_i + 1/N + (i/N)*c
    # 
    # Breakpoint: Left_i(c) = Right_j(c) for i != j
    # x_i + (i/N)*c = x_j + 1/N + (j/N)*c
    # (i-j)/N * c = x_j + 1/N - x_i
    # c = N*(x_j + 1/N - x_i) / (i-j)
    
    # Also: Left_i(c) = Left_j(c) -> same slope ordering, but they cross at:
    # x_i + (i/N)*c = x_j + (j/N)*c
    # c = N*(x_j - x_i)/(i-j)
    
    # Collect all breakpoints in [0, 1]
    breakpoints = [0.0, 1.0]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                # Left_i = Right_j
                denom = (i - j) / N
                if abs(denom) > 1e-12:
                    c_val = (x[j] + 1.0/N - x[i]) / denom
                    if 0.0 < c_val < 1.0:
                        breakpoints.append(c_val)
    
    breakpoints = sorted(set([round(b, 10) for b in breakpoints]))
    
    # At each interval between consecutive breakpoints, the ordering of
    # intervals is constant, so the union length is linear in c.
    # We can evaluate at the midpoint of each sub-interval.
    
    if len(breakpoints) < 2:
        # No breakpoints, just evaluate at c=0.5
        c = 0.5
        lefts = x + c * np.arange(N) / N
        rights = lefts + 1.0/N
        # Compute union length
        order = np.argsort(lefts)
        lefts_sorted = lefts[order]
        rights_sorted = rights[order]
        union_len = 0.0
        current_right = rights_sorted[0]
        for k in range(1, N):
            if lefts_sorted[k] > current_right:
                union_len += current_right - lefts_sorted[0] if k == 1 else current_right - lefts_sorted[k-1]
                lefts_sorted = lefts_sorted[k:]
                rights_sorted = rights_sorted[k:]
                current_right = rights_sorted[0]
            else:
                current_right = max(current_right, rights_sorted[k])
        union_len += current_right - lefts_sorted[0]
        return union_len
    
    # Evaluate union length at midpoints of each sub-interval
    total_length = 0.0
    for idx in range(len(breakpoints) - 1):
        c_mid = 0.5 * (breakpoints[idx] + breakpoints[idx + 1])
        width = breakpoints[idx + 1] - breakpoints[idx]
        
        lefts = x + c_mid * np.arange(N) / N
        rights = lefts + 1.0/N
        
        # Compute union length of intervals
        order = np.argsort(lefts)
        lefts_sorted = lefts[order]
        rights_sorted = rights[order]
        
        union_len = 0.0
        current_right = rights_sorted[0]
        for k in range(1, N):
            if lefts_sorted[k] > current_right:
                union_len += current_right - lefts_sorted[0] if k == 1 else current_right - lefts_sorted[k-1]
                lefts_sorted = lefts_sorted[k:]
                rights_sorted = rights_sorted[k:]
                current_right = rights_sorted[0]
            else:
                current_right = max(current_right, rights_sorted[k])
        union_len += current_right - lefts_sorted[0]
        
        total_length += union_len * width
    
    return total_length

def _keich_offsets(N=8):
    """Keich binary construction for N=2^k."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich

def _optimize_1d(x0, deadline):
    """
    Optimize 1D positions to minimize the integrated union length.
    """
    N = 8
    
    def cost(x):
        return _compute_1d_union_length(x, N)
    
    best_x = x0.copy()
    best_cost = cost(x0)
    
    # Multiple restarts with Nelder-Mead
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb
            x_init = x0 + 0.1 * np.random.randn(N)
        
        try:
            res = minimize(cost, x_init, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x, best_cost

def _compute_volume_from_1d(x, y, N=8):
    """
    Estimate 3D volume from 1D union lengths.
    
    The 3D volume is approximately the integral over c of:
    (union_length_x(c) * union_length_y(c))
    
    We compute this exactly using the same breakpoint analysis.
    """
    # Get all breakpoints from both x and y
    all_breakpoints = [0.0, 1.0]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                denom = (i - j) / N
                if abs(denom) > 1e-12:
                    c_val = (x[j] + 1.0/N - x[i]) / denom
                    if 0.0 < c_val < 1.0:
                        all_breakpoints.append(c_val)
                    c_val = (y[j] + 1.0/N - y[i]) / denom
                    if 0.0 < c_val < 1.0:
                        all_breakpoints.append(c_val)
    
    all_breakpoints = sorted(set([round(b, 10) for b in all_breakpoints]))
    
    if len(all_breakpoints) < 2:
        c = 0.5
        lx = x + c * np.arange(N) / N
        rx = lx + 1.0/N
        ly = y + c * np.arange(N) / N
        ry = ly + 1.0/N
        
        # Union length x
        ox = np.argsort(lx)
        lx_s, rx_s = lx[ox], rx[ox]
        ulx = 0.0
        cr = rx_s[0]
        for k in range(1, N):
            if lx_s[k] > cr:
                ulx += cr - lx_s[0] if k == 1 else cr - lx_s[k-1]
                lx_s, rx_s = lx_s[k:], rx_s[k:]
                cr = rx_s[0]
            else:
                cr = max(cr, rx_s[k])
        ulx += cr - lx_s[0]
        
        # Union length y
        oy = np.argsort(ly)
        ly_s, ry_s = ly[oy], ry[oy]
        uly = 0.0
        cr = ry_s[0]
        for k in range(1, N):
            if ly_s[k] > cr:
                uly += cr - ly_s[0] if k == 1 else cr - ly_s[k-1]
                ly_s, ry_s = ly_s[k:], ry_s[k:]
                cr = ry_s[0]
            else:
                cr = max(cr, ry_s[k])
        uly += cr - ly_s[0]
        
        return ulx * uly
    
    total_volume = 0.0
    for idx in range(len(all_breakpoints) - 1):
        c_mid = 0.5 * (all_breakpoints[idx] + all_breakpoints[idx + 1])
        width = all_breakpoints[idx + 1] - all_breakpoints[idx]
        
        lx = x + c_mid * np.arange(N) / N
        rx = lx + 1.0/N
        ly = y + c_mid * np.arange(N) / N
        ry = ly + 1.0/N
        
        # Union length x
        ox = np.argsort(lx)
        lx_s, rx_s = lx[ox], rx[ox]
        ulx = 0.0
        cr = rx_s[0]
        for k in range(1, N):
            if lx_s[k] > cr:
                ulx += cr - lx_s[0] if k == 1 else cr - lx_s[k-1]
                lx_s, rx_s = lx_s[k:], rx_s[k:]
                cr = rx_s[0]
            else:
                cr = max(cr, rx_s[k])
        ulx += cr - lx_s[0]
        
        # Union length y
        oy = np.argsort(ly)
        ly_s, ry_s = ly[oy], ry[oy]
        uly = 0.0
        cr = ry_s[0]
        for k in range(1, N):
            if ly_s[k] > cr:
                uly += cr - ly_s[0] if k == 1 else cr - ly_s[k-1]
                ly_s, ry_s = ly_s[k:], ry_s[k:]
                cr = ry_s[0]
            else:
                cr = max(cr, ry_s[k])
        uly += cr - ly_s[0]
        
        total_volume += ulx * uly * width
    
    # The actual bounding box area multiplies this
    # X0 = min(x), X1 = max(x + (i+1)/N)
    # Y0 = min(y), Y1 = max(y + (j+1)/N)
    X0 = np.min(x)
    X1 = np.max(x + (np.arange(N) + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (np.arange(N) + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    
    return total_volume * box_area

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    """
    deadline = time.time() + 580  # Safety margin before 600s limit
    
    N = 8
    keich = _keich_offsets(N)
    
    # Optimize x and y independently using exact 1D union length
    x_opt, x_cost = _optimize_1d(keich.copy(), deadline)
    y_opt, y_cost = _optimize_1d(keich.copy(), deadline)
    
    # Build full 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
