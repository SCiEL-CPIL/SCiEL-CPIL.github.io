# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area(x, n=64):
    """
    Compute the union area of n triangles using exact piecewise-linear analysis.
    L_i(y) = x_i + a_i*y, a_i = (i+1)/n
    w(y) = (1-y)/n
    Union length at height y: w + sum over consecutive pairs min(gap, w)
    """
    a = (np.arange(n) + 1) / n
    
    # Find all pairwise intersection points y where L_i(y) = L_j(y)
    # x_i + a_i*y = x_j + a_j*y => y = (x_j - x_i)/(a_i - a_j)
    y_bounds = [0.0, 1.0]
    for i in range(n):
        for j in range(i+1, n):
            da = a[i] - a[j]
            if abs(da) < 1e-15:
                continue
            y_int = (x[j] - x[i]) / da
            if 0.0 < y_int < 1.0:
                y_bounds.append(y_int)
    
    y_bounds = np.array(sorted(set(np.round(y_bounds, 12))))
    
    total_area = 0.0
    for k in range(len(y_bounds) - 1):
        y_lo = y_bounds[k]
        y_hi = y_bounds[k+1]
        if y_hi - y_lo < 1e-14:
            continue
        
        # Midpoint of this segment to determine ordering
        y_mid = 0.5 * (y_lo + y_hi)
        L_mid = x + a * y_mid
        order = np.argsort(L_mid)
        
        # Get ordered L values at endpoints
        L_lo = x[order] + a[order] * y_lo
        L_hi = x[order] + a[order] * y_hi
        
        # Gaps between consecutive lines in this segment
        gap_lo = L_lo[1:] - L_lo[:-1]
        gap_hi = L_hi[1:] - L_hi[:-1]
        
        # w at endpoints
        w_lo = (1.0 - y_lo) / n
        w_hi = (1.0 - y_hi) / n
        
        # Union length = w + sum(min(gap, w))
        # Since gap and w are both linear in y, min(gap, w) is piecewise linear
        # But within a segment, the ordering of lines is fixed, and each gap
        # is linear. min(gap, w) changes only if gap and w cross within the segment.
        # For simplicity and correctness, evaluate at midpoint (trapezoidal approx 
        # within each segment is exact if we handle the min properly)
        
        # Actually, let's be more careful. Within a segment, each gap_j(y) is linear.
        # w(y) is also linear. min(gap_j(y), w(y)) is the minimum of two linear functions,
        # which is piecewise linear with at most one breakpoint per gap.
        # For numerical robustness, let's just use the midpoint value times segment width.
        # This is an approximation but should be very accurate for small segments.
        
        y_mid = 0.5 * (y_lo + y_hi)
        L_mid = x[order] + a[order] * y_mid
        gap_mid = L_mid[1:] - L_mid[:-1]
        w_mid = (1.0 - y_mid) / n
        
        union_mid = w_mid + np.sum(np.minimum(gap_mid, w_mid))
        total_area += union_mid * (y_hi - y_lo)
    
    return total_area / 1.0  # integral over [0,1]

def _compute_area_fast(x, n=64, M=20000):
    """
    Fast quadrature-based area computation (matches the evaluator exactly).
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return float(area)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    
    # Initial guess: x_i = -a_i * 0.5 (centered)
    a = (np.arange(n) + 1) / n
    x0 = -a * 0.5
    
    best_x = x0.copy()
    best_area = _compute_area_fast(x0, n, M)
    
    # Try several initial configurations
    inits = [
        np.zeros(n),
        -a * 0.5,
        -a * (1.0/3.0),
        -a * (2.0/3.0),
        a * 0.5,
        -a * 0.25,
        -a * 0.75,
    ]
    for init in inits:
        area_init = _compute_area_fast(init, n, M)
        if area_init < best_area:
            best_area = area_init
            best_x = init.copy()
    
    # Subgradient descent using the fast area function
    # The area is a piecewise-linear (convex) function of x, so subgradient methods work well.
    
    def _area_subgrad(x, n=64, M=20000):
        """Compute area and subgradient via finite differences or analytic subgradient."""
        m = np.arange(M)
        y = (m + 0.5) / M
        a = (np.arange(n) + 1) / n
        w = (1.0 - y) / n
        
        L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        
        # Track which line is which in the sorted order
        # For subgradient: d(area)/d(x_i) = integral of indicator that line i is at boundary of union
        # More precisely, the union length = w + sum_j min(gap_j, w)
        # where gaps are between consecutive lines in sorted order.
        # The subgradient with respect to x_i involves how x_i affects each gap.
        
        # Simpler: use the fact that area is convex and use finite differences for gradient
        eps = 1e-7
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (_compute_area_fast(x_plus, n, M) - _compute_area_fast(x_minus, n, M)) / (2*eps)
        
        return _compute_area_fast(x, n, M), grad
    
    # Use Nelder-Mead or simple subgradient descent
    # Given the expense of gradient computation, use a coordinate-wise approach
    
    # Coordinate descent with line search
    x = best_x.copy()
    current_area = best_area
    
    for iteration in range(200):
        if time.time() > deadline - 30:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 30:
                break
            
            # Try small perturbations in coordinate i
            best_local = current_area
            best_delta = 0.0
            for delta in [-0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05]:
                x_trial = x.copy()
                x_trial[i] += delta
                area_trial = _compute_area_fast(x_trial, n, M)
                if area_trial < best_local:
                    best_local = area_trial
                    best_delta = delta
            
            if best_delta != 0.0:
                x[i] += best_delta
                current_area = best_local
                improved = True
        
        if not improved:
            # Try smaller steps
            improved_small = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                for delta in [-0.002, -0.001, 0.001, 0.002]:
                    x_trial = x.copy()
                    x_trial[i] += delta
                    area_trial = _compute_area_fast(x_trial, n, M)
                    if area_trial < current_area - 1e-10:
                        x[i] += delta
                        current_area = area_trial
                        improved_small = True
                        break
            if not improved_small:
                break
        
        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()
    
    # Final polish with scipy if available
    if time.time() < deadline - 60:
        try:
            from scipy.optimize import minimize
            def objective(x):
                return _compute_area_fast(x, n, M)
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun
        except Exception:
            pass
    
    # Validation
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    if best_area <= 0:
        return np.zeros(n, dtype=np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
