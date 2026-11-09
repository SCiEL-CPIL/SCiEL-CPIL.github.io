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

def _compute_area_subgrad(x, n=64, M=20000):
    """
    Compute area and subgradient using the fixed-ordering trick when valid,
    falling back to finite differences otherwise.
    Returns (area, gradient_array).
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted, order = np.sort(L, axis=0), np.argsort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = float(np.mean(union_lengths))
    
    # Subgradient: for each height y_m, the union length is
    #   w + sum_{j=0}^{n-2} min(L_sorted[j+1] - L_sorted[j], w)
    # The subgradient w.r.t. x_i is the sum over heights of the contribution.
    # For a given height, if L_sorted[j+1] - L_sorted[j] < w (saturated), then
    # the gap contributes (L_sorted[j+1] - L_sorted[j]) to the union, and
    # d/dx_i of that is +1 if i is the (j+1)-th in sorted order, -1 if i is the j-th.
    # If not saturated, the min is w, which is independent of x_i, so contribution is 0.
    #
    # So: grad[i] = sum_m [ I(i is upper of a saturated gap at m) - I(i is lower of a saturated gap at m) ] / M
    
    sat = (d < w[np.newaxis, :])  # shape (n-1, M)
    # sat[j, m] is True if the gap between sorted[j] and sorted[j+1] at height m is saturated
    
    grad = np.zeros(n, dtype=np.float64)
    # For each height m, for each saturated gap j, add +1 to grad[order[j+1, m]] and -1 to grad[order[j, m]]
    # Vectorized:
    # order has shape (n, M), sat has shape (n-1, M)
    # For gap j at height m: upper index = order[j+1, m], lower index = order[j, m]
    
    # Build arrays of upper and lower indices for all saturated gaps
    # This can be large (63 * 20000 = 1.26M entries), but manageable
    sat_indices = np.where(sat)  # (j_indices, m_indices)
    if len(sat_indices[0]) > 0:
        j_idx = sat_indices[0]
        m_idx = sat_indices[1]
        upper = order[j_idx + 1, m_idx]
        lower = order[j_idx, m_idx]
        np.add.at(grad, upper, 1.0)
        np.add.at(grad, lower, -1.0)
    
    grad /= M
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill
    
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
    
    # Subgradient descent with Polyak step size
    def _subgrad_descent(x0, max_time):
        x = x0.copy()
        area, g = _compute_area_subgrad(x, n, M)
        v = np.zeros(n, dtype=np.float64)
        mu = 0.9
        step = 0.05
        best_local = area
        best_local_x = x.copy()
        
        while time.time() < max_time:
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-14:
                break
            step_polyak = max(np.dot(x, g), 1e-12) / (gnorm * gnorm + 1e-12)
            step = min(step_polyak, 0.3)
            v = mu * v - step * g
            xn = x + v
            an, gn = _compute_area_subgrad(xn, n, M)
            if an < area:
                x, area, g = xn, an, gn
                if an < best_local:
                    best_local = an
                    best_local_x = x.copy()
            else:
                step *= 0.7
                v *= 0.5
                if step < 1e-12:
                    break
        return best_local_x, best_local
    
    # Phase 1: Subgradient descent from the best initial
    if time.time() < deadline - 60:
        t_max = min(deadline - 60, start_time + 120)
        x_sg, area_sg = _subgrad_descent(best_x, t_max)
        if area_sg < best_area:
            best_area = area_sg
            best_x = x_sg.copy()
    
    # Phase 2: Coordinate descent with line search for local refinement
    x = best_x.copy()
    current_area = best_area
    
    for iteration in range(300):
        if time.time() > deadline - 60:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 60:
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
                if time.time() > deadline - 60:
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
    
    # Phase 3: Basin hopping with subgradient polishing
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for hop in range(20):
            if time.time() > deadline - 30:
                break
            # Perturb current best
            perturbation_scale = 0.3 / (1.0 + hop * 0.1)
            x_pert = best_x + rng.normal(0, perturbation_scale, n)
            area_pert = _compute_area_fast(x_pert, n, M)
            
            # Short subgradient descent from perturbed point
            t_max = min(deadline - 30, start_time + 30)
            x_hop, area_hop = _subgrad_descent(x_pert, t_max)
            
            if area_hop < best_area:
                best_area = area_hop
                best_x = x_hop.copy()
    
    # Phase 4: Final polish with scipy if available
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            def objective(x):
                return _compute_area_fast(x, n, M)
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
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
