import numpy as np
import time
from scipy.optimize import minimize

def _union_area(rects):
    """
    Compute the area of the union of axis-aligned rectangles.
    rects: list of (x1, y1, x2, y2)
    """
    if not rects:
        return 0.0
    
    # Collect x-coordinates
    xs = set()
    for x1, y1, x2, y2 in rects:
        xs.add(x1)
        xs.add(x2)
    xs = sorted(list(xs))
    
    if len(xs) < 2:
        return 0.0
        
    area = 0.0
    for i in range(len(xs) - 1):
        xa = xs[i]
        xb = xs[i + 1]
        if xb <= xa:
            continue
            
        # Find rectangles covering [xa, xb]
        # A rectangle (x1, y1, x2, y2) covers [xa, xb] if x1 <= xa and x2 >= xb
        y_intervals = []
        for x1, y1, x2, y2 in rects:
            if x1 <= xa + 1e-12 and x2 >= xb - 1e-12:
                y_intervals.append((y1, y2))
                
        if not y_intervals:
            continue
            
        # Compute length of union of y_intervals
        y_intervals.sort(key=lambda t: t[0])
        current_start = y_intervals[0][0]
        current_end = y_intervals[0][1]
        y_union_len = 0.0
        
        for start, end in y_intervals[1:]:
            if start > current_end + 1e-12:
                y_union_len += current_end - current_start
                current_start = start
                current_end = end
            else:
                if end > current_end:
                    current_end = end
        y_union_len += current_end - current_start
        
        area += (xb - xa) * y_union_len
        
    return area

def _get_breakpoints(X, Y, N=8):
    """
    Get all c in [0, 1] where the relative order of x-edges or y-edges changes.
    """
    breakpoints = set([0.0, 1.0])
    
    # X edges: X_i + i/N * c and X_i + i/N * c + 1/N
    # Y edges: Y_j + j/N * c and Y_j + j/N * c + 1/N
    
    def get_edges_coords(coords, N):
        edges = []
        for i in range(N):
            edges.append((coords[i], i, 0)) # Left edge, index i, type 0
            edges.append((coords[i] + 1.0/N, i, 1)) # Right edge, index i, type 1
        return edges

    x_edges = get_edges_coords(X, N)
    y_edges = get_edges_coords(Y, N)
    
    # Compare all pairs of x-edges
    for i in range(len(x_edges)):
        for j in range(i + 1, len(x_edges)):
            xi, ii, ti = x_edges[i]
            xj, jj, tj = x_edges[j]
            # xi + ii/N * c = xj + jj/N * c
            # c * (ii - jj) / N = xj - xi
            denom = (ii - jj) / N
            if abs(denom) > 1e-12:
                c = (xj - xi) / denom
                if 0.0 < c < 1.0:
                    breakpoints.add(c)
                    
    # Compare all pairs of y-edges
    for i in range(len(y_edges)):
        for j in range(i + 1, len(y_edges)):
            yi, ii, ti = y_edges[i]
            yj, jj, tj = y_edges[j]
            denom = (ii - jj) / N
            if abs(denom) > 1e-12:
                c = (yj - yi) / denom
                if 0.0 < c < 1.0:
                    breakpoints.add(c)
                    
    return sorted(breakpoints)

def _compute_volume(X, Y, N=8):
    """
    Compute the exact union volume.
    """
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        return 1e10
        
    breakpoints = _get_breakpoints(X, Y, N)
    
    total_volume = 0.0
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        if c_end - c_start < 1e-12:
            continue
            
        # Sample 3 points to determine the quadratic polynomial A(c)
        c1 = c_start
        c2 = 0.5 * (c_start + c_end)
        c3 = c_end
        
        def get_rects(c):
            rects = []
            for i in range(N):
                for j in range(N):
                    x1 = X[i] + (i / N) * c
                    x2 = x1 + 1.0 / N
                    y1 = Y[j] + (j / N) * c
                    y2 = y1 + 1.0 / N
                    rects.append((x1, y1, x2, y2))
            return rects
            
        rects1 = get_rects(c1)
        rects2 = get_rects(c2)
        rects3 = get_rects(c3)
        
        A1 = _union_area(rects1)
        A2 = _union_area(rects2)
        A3 = _union_area(rects3)
        
        # A(c) = a c^2 + b c + d
        # We have 3 points.
        # Solve for a, b, d.
        # Or use Lagrange interpolation formula for the integral.
        # Integral from c1 to c3 of A(c) dc.
        # Simpson's rule is exact for quadratics.
        # Integral = (c3 - c1) / 6 * (A1 + 4*A2 + A3)
        
        total_volume += (c3 - c1) / 6.0 * (A1 + 4.0 * A2 + A3)
        
    return total_volume

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570 # Leave 30s margin
    
    N = 8
    
    def objective(params):
        X = params[:N]
        Y = params[N:]
        return _compute_volume(X, Y, N)
        
    # Initial guesses
    inits = []
    # Random inits
    rng = np.random.default_rng(12345)
    for _ in range(20):
        X = rng.uniform(-0.5, 0.5, N)
        Y = rng.uniform(-0.5, 0.5, N)
        inits.append(np.concatenate([X, Y]))
        
    # Structured inits
    X0 = np.zeros(N)
    Y0 = np.zeros(N)
    inits.append(np.concatenate([X0, Y0]))
    
    X1 = np.arange(N) / N
    Y1 = np.arange(N) / N
    inits.append(np.concatenate([X1, Y1]))
    
    X2 = -np.arange(N) / N
    Y2 = -np.arange(N) / N
    inits.append(np.concatenate([X2, Y2]))
    
    best_params = None
    best_vol = float('inf')
    
    # Phase 1: Nelder-Mead from multiple starts
    for init in inits:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        best_params = np.zeros(2 * N)
        
    # Phase 2: Powell's method for refinement
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_params, method='Powell',
                           options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Another Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    X = best_params[:N]
    Y = best_params[N:]
    
    # Build positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X[i_flat]
    base_y = Y[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        positions = np.zeros((64, 2))
        
    return positions
