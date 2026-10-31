import numpy as np
def construct():
    N = 100
    target_sum = N / 2.0
    
    def _max_overlap(h):
        one_minus_h = 1.0 - h
        max_ov = 0.0
        for s in range(N):
            shifted = np.roll(one_minus_h, s)
            ov = np.dot(h, shifted)
            if ov > max_ov:
                max_ov = ov
        return max_ov

    def _construct_h(w, c):
        i = np.arange(N)
        if w < 1e-6:
            w = 1e-6
        h = 0.5 + 0.5 * np.tanh((i - c) / w)
        return h

    def _solve_c(w):
        # Binary search for c
        lo = -10.0 * w # Extend range to be safe
        hi = N + 10.0 * w
        
        # Actually, c should be within [0, N] roughly for the step to be inside.
        # If c is far outside, sum is 0 or N.
        # We need sum = N/2.
        # Sum is decreasing with c.
        # Sum(c -> -inf) = N
        # Sum(c -> +inf) = 0
        # So c is in [0, N] roughly.
        
        # Let's use a wider range for binary search to be safe
        lo = -N
        hi = 2 * N
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            h_mid = _construct_h(w, mid)
            s_mid = np.sum(h_mid)
            if s_mid > target_sum:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def _objective(w):
        c = _solve_c(w)
        h = _construct_h(w, c)
        # Correct sum exactly
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-9:
             # Distribute diff
             # Since h is in [0,1], we can add diff/N
             # But this might push out of bounds if w is small.
             # For w > 0.5, h is mostly 0 or 1, transition is small.
             # Adding diff/N is small.
             h = h + diff / N
             h = np.clip(h, 0.0, 1.0)
             # Re-correct if clip changed sum? 
             # The binary search should be very accurate.
        return _max_overlap(h)

    # Grid search over log(w)
    # w in [0.1, 100]
    # log(w) in [-2.3, 4.6]
    log_w_grid = np.linspace(-2.0, 4.5, 500)
    
    best_w = None
    best_val = float('inf')
    
    for lw in log_w_grid:
        w = np.exp(lw)
        val = _objective(w)
        if val < best_val:
            best_val = val
            best_w = w
            
    # Refine around best_w
    # Use minimize_scalar on the refined region
    # Or just a finer grid
    if best_w is not None:
        # Find index
        idx = np.argmin([_objective(np.exp(lw)) for lw in log_w_grid]) # This is slow, re-calculating
        # Better: just use the stored best_w
        pass

    # Let's do a proper refinement using minimize_scalar on the best region
    # Determine bounds for refinement
    # Find neighbors in grid
    # ...
    
    # Simpler: Use minimize_scalar on the whole range but with higher precision?
    # The grid search found the basin.
    # Let's do a local optimization around best_w.
    
    from scipy.optimize import minimize_scalar
    
    # Define a wrapper for minimize_scalar that takes log_w
    def obj_log_w(lw):
        return _objective(np.exp(lw))
        
    # Bounds for refinement: +/- 0.5 in log scale around best
    best_lw = np.log(best_w)
    bounds = (best_lw - 0.5, best_lw + 0.5)
    
    res = minimize_scalar(obj_log_w, bounds=bounds, method='bounded', options={'xatol': 1e-6})
    
    if res.success:
        best_w = np.exp(res.x)
        
    c = _solve_c(best_w)
    h = _construct_h(best_w, c)
    
    # Final sum correction
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Iterative fix
        for _ in range(10):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12: break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = min(1.0, h[idx] + step)
            else:
                mask = h > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = max(0.0, h[idx] + step)

    return h
