import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    N = 8
    w = 1.0 / N
    slopes = np.arange(N, dtype=float) / N
    
    def _compute_3d_volume(x, y):
        """
        Compute exact 3D union volume of 64 sheared tubes.
        Tube (i,j) at height c: [x_i + c*i/N, x_i + c*i/N + w] x [y_j + c*j/N, y_j + c*j/N + w]
        Volume = integral_0^1 (area of union of rectangles at c) dc
        """
        # Find all breakpoints in c where the combinatorial structure of the union changes
        bps = {0.0, 1.0}
        
        # Breakpoints from x-dimension: x_i + c*s_i + d1 = x_j + c*s_j + d2
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = slopes[i] - slopes[j]
                if abs(denom) < 1e-15:
                    continue
                for d1 in (0.0, w):
                    for d2 in (0.0, w):
                        c_val = (x[j] - x[i] + d2 - d1) / denom
                        if -1e-12 < c_val < 1.0 + 1e-12:
                            c_val = max(0.0, min(1.0, c_val))
                            bps.add(c_val)
        
        # Breakpoints from y-dimension: y_j + c*s_j + d1 = y_i + c*s_i + d2
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = slopes[i] - slopes[j]
                if abs(denom) < 1e-15:
                    continue
                for d1 in (0.0, w):
                    for d2 in (0.0, w):
                        c_val = (y[j] - y[i] + d2 - d1) / denom
                        if -1e-12 < c_val < 1.0 + 1e-12:
                            c_val = max(0.0, min(1.0, c_val))
                            bps.add(c_val)
        
        bps = sorted(bps)
        
        total_vol = 0.0
        for k in range(len(bps) - 1):
            c0 = bps[k]
            c1 = bps[k + 1]
            h = c1 - c0
            if h < 1e-12:
                continue
            
            # Sample at midpoint for robustness
            c_mid = 0.5 * (c0 + c1)
            
            # Compute union area at c_mid
            # Rectangles: [x_i + c_mid*s_i, x_i + c_mid*s_i + w] x [y_j + c_mid*s_j, y_j + c_mid*s_j + w]
            x_lefts = x + c_mid * slopes
            x_rights = x_lefts + w
            y_lefts = y + c_mid * slopes
            y_rights = y_lefts + w
            
            # Compute union area using sweep line
            # Merge x-intervals
            x_order = np.argsort(x_lefts)
            x_lefts_s = x_lefts[x_order]
            x_rights_s = x_rights[x_order]
            
            x_segments = []
            cur_l = x_lefts_s[0]
            cur_r = x_rights_s[0]
            for idx in range(1, N):
                if x_lefts_s[idx] <= cur_r + 1e-14:
                    cur_r = max(cur_r, x_rights_s[idx])
                else:
                    x_segments.append((cur_l, cur_r))
                    cur_l = x_lefts_s[idx]
                    cur_r = x_rights_s[idx]
            x_segments.append((cur_l, cur_r))
            
            # Merge y-intervals
            y_order = np.argsort(y_lefts)
            y_lefts_s = y_lefts[y_order]
            y_rights_s = y_rights[y_order]
            
            y_segments = []
            cur_l = y_lefts_s[0]
            cur_r = y_rights_s[0]
            for idx in range(1, N):
                if y_lefts_s[idx] <= cur_r + 1e-14:
                    cur_r = max(cur_r, y_rights_s[idx])
                else:
                    y_segments.append((cur_l, cur_r))
                    cur_l = y_lefts_s[idx]
                    cur_r = y_rights_s[idx]
            y_segments.append((cur_l, cur_r))
            
            # Compute area = sum over x_segments of (width * union_y_length_within_x_segment)
            # Actually, the area of union of rectangles is not simply separable.
            # We need to compute the actual union area.
            # Use the fact that between breakpoints, the union area is a quadratic polynomial in c.
            # So we evaluate at c0, c_mid, c1 and use Simpson's rule or quadratic interpolation.
            
            # Let's compute union area at c0, c_mid, c1
            def union_area(c_val):
                xl = x + c_val * slopes
                xr = xl + w
                yl = y + c_val * slopes
                yr = yl + w
                
                # Get merged x intervals
                xo = np.argsort(xl)
                xls = xl[xo]
                xrs = xr[xo]
                xsegs = []
                cl = xls[0]
                cr = xrs[0]
                for idx in range(1, N):
                    if xls[idx] <= cr + 1e-14:
                        cr = max(cr, xrs[idx])
                    else:
                        xsegs.append((cl, cr))
                        cl = xls[idx]
                        cr = xrs[idx]
                xsegs.append((cl, cr))
                
                # Get merged y intervals
                yo = np.argsort(yl)
                yls = yl[yo]
                yrs = yr[yo]
                ysegs = []
                cl = yls[0]
                cr = yrs[0]
                for idx in range(1, N):
                    if yls[idx] <= cr + 1e-14:
                        cr = max(cr, yrs[idx])
                    else:
                        ysegs.append((cl, cr))
                        cl = yls[idx]
                        cr = yrs[idx]
                ysegs.append((cl, cr))
                
                # Compute union area: for each pair of x-seg and y-seg, check if they come from same tube
                # Actually, the union of rectangles is not a Cartesian product of merged intervals.
                # We need to compute the area of the union of N*N rectangles.
                # This is O(N^3) per evaluation, which is fine for N=8.
                
                # Alternative: use inclusion-exclusion or just compute the area directly.
                # For N=8, we have 64 rectangles. We can use a grid-free approach.
                # The area of union of axis-aligned rectangles can be computed by sweep line in O(n log n).
                
                # For simplicity and correctness, let's use the fact that between breakpoints,
                # the set of rectangles that overlap in x and y is constant.
                # Actually, the combinatorial structure of the union changes only at breakpoints.
                # So the area function is piecewise quadratic.
                
                # Let's just compute the area by summing over all pairs and using inclusion-exclusion?
                # No, that's O(2^64) which is infeasible.
                
                # Better: sweep line algorithm for union area of rectangles.
                # Events: x-coordinate of left and right edges.
                # For each x-interval between events, compute the union length in y.
                
                events = []
                for i in range(N):
                    for j in range(N):
                        xl = x[i] + c_val * slopes[i]
                        xr = xl + w
                        yl = y[j] + c_val * slopes[j]
                        yr = yl + w
                        events.append((xl, 1, yl, yr))
                        events.append((xr, -1, yl, yr))
                
                events.sort()
                
                active_y = []
                area = 0.0
                prev_x = None
                
                for ev in events:
                    ex, typ, yl, yr = ev
                    if prev_x is not None and ex > prev_x:
                        # Compute union length of active y-intervals
                        if active_y:
                            active_y.sort()
                            cur_l, cur_r = active_y[0]
                            total_len = 0.0
                            for idx in range(1, len(active_y)):
                                l, r = active_y[idx]
                                if l <= cur_r + 1e-14:
                                    cur_r = max(cur_r, r)
                                else:
                                    total_len += cur_r - cur_l
                                    cur_l, cur_r = l, r
                            total_len += cur_r - cur_l
                            area += total_len * (ex - prev_x)
                    
                    if typ == 1:
                        active_y.append((yl, yr))
                    else:
                        active_y.remove((yl, yr))
                    
                    prev_x = ex
                
                return area
            
            A0 = union_area(c0)
            Am = union_area(c_mid)
            A1 = union_area(c1)
            
            # Simpson's rule
            total_vol += h / 6.0 * (A0 + 4 * Am + A1)
        
        return total_vol
    
    # Optimize 14 variables: x[1..7] and y[1..7]. x[0]=0, y[0]=0.
    def objective(params):
        x = np.zeros(N)
        y = np.zeros(N)
        x[1:] = params[:N-1]
        y[1:] = params[N-1:]
        return _compute_3d_volume(x, y)
    
    best_params = None
    best_vol = float('inf')
    
    # Strategy 1: Differential Evolution for global search
    bounds = [(-1.0, 1.0)] * (2 * (N - 1))
    
    try:
        if time.time() < deadline - 150:
            de_res = differential_evolution(
                objective, 
                bounds, 
                maxiter=30, 
                popsize=10, 
                tol=0.01, 
                seed=42,
                polish=False
            )
            if de_res.fun < best_vol:
                best_vol = de_res.fun
                best_params = de_res.x
    except Exception:
        pass
    
    # Strategy 2: Local polish with Nelder-Mead
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
    
    # Symmetric guesses
    for delta in [0.1, 0.2, 0.5, 1.0]:
        x_guess = np.linspace(-delta, delta, N)
        x_guess[0] = 0.0
        p = np.concatenate([x_guess[1:], x_guess[1:]])
        candidates.append(p)
    
    # Random guesses
    rng = np.random.default_rng(123)
    for _ in range(5):
        p = rng.uniform(-0.5, 0.5, 2*(N-1))
        candidates.append(p)
    
    for p0 in candidates:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                objective, 
                p0, 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Construct final x, y
    if best_params is None:
        best_params = np.zeros(2*(N-1))
        
    x = np.zeros(N)
    y = np.zeros(N)
    x[1:] = best_params[:N-1]
    y[1:] = best_params[N-1:]
    
    # Verify
    final_vol = _compute_3d_volume(x, y)
    if final_vol >= 1e5 or not np.isfinite(final_vol) or final_vol <= 0:
        # Fallback to simple symmetric
        x = np.linspace(-0.5, 0.5, N)
        y = np.linspace(-0.5, 0.5, N)
        x[0] = 0.0
        y[0] = 0.0
        
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x[i_flat]
    base_y = y[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
