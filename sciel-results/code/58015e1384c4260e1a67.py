<<<<<<< SEARCH
def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize Kakeya needle triangle positions to minimize union area.
    Uses coordinate descent with subgradient-based 1D optimization.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    n = 64
    x = _keich_initial(n)
    best_x = x.copy()
    best_area = _area_fast(x, n)
    
    # Coordinate descent
    max_passes = 20
    tol = 1e-12
    
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # 1D optimization for coordinate i using subgradient bisection
            # The area is convex in x_i, so we can use subgradient to find minimum
            
            # Find a bracket for the optimal x_i
            # Start with current x[i] and expand
            lo = x[i] - 0.1
            hi = x[i] + 0.1
            
            # Evaluate subgradient at lo and hi
            sg_lo = _subgrad_xi(x, i, n)
            # We need to adjust x[i] to lo and evaluate
            x[i] = lo
            sg_lo = _subgrad_xi(x, i, n)
            x[i] = hi
            sg_hi = _subgrad_xi(x, i, n)
            
            # If sg_lo > 0, minimum is to the left of lo
            # If sg_hi < 0, minimum is to the right of hi
            # Otherwise, minimum is in [lo, hi]
            
            # Bisection on subgradient
            for _ in range(30):  # 30 iterations of bisection
                if time.time() > deadline:
                    break
                
                mid = (lo + hi) / 2
                x[i] = mid
                sg_mid = _subgrad_xi(x, i, n)
                
                if sg_mid > 0:
                    hi = mid
                else:
                    lo = mid
            
            # Set x[i] to the midpoint
            x[i] = (lo + hi) / 2
            new_area = _area_fast(x, n)
            
            if new_area < best_area - 1e-15:
                best_area = new_area
                best_x = x.copy()
                improved = True
            else:
                # Revert if no improvement
                x[i] = best_x[i]
        
        if not improved:
            break
    
    # Return best result
    final_area = _area_fast(best_x, n)
    if np.isfinite(final_area) and final_area > 0:
        return best_x
    return _keich_initial(n)
=======
def _smooth_area(x, n=64, n_heights=20000, beta=500.0):
    """Compute smooth approximation of union area using soft-min."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Soft-min: min(gap, w) ≈ -1/beta * log(1 + exp(-beta*gap)) + w - 1/beta*log(1+exp(-beta*w))
    # Actually simpler: softmin(gap, w) = -(1/beta)*log(exp(-beta*gap) + exp(-beta*w))
    # But we want min(gap, w), so:
    # softmin(a,b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # For numerical stability:
    # softmin(gap, w) = -1/beta * log(exp(-beta*(gap - w)) + 1) + w  (when w is the max)
    # Actually let's use: softmin(gap, w) = w - 1/beta * log1p(exp(-beta*(gap - w)))
    # Wait: min(gap, w). If gap < w, min = gap. If gap > w, min = w.
    # softmin = -1/beta * log(exp(-beta*gap) + exp(-beta*w))
    # = -1/beta * log(exp(-beta*w) * (exp(-beta*(gap-w)) + 1))
    # = w - 1/beta * log(1 + exp(-beta*(gap - w)))
    
    # For large beta, this approximates min(gap, w)
    # Use log1p for numerical stability
    softmin_vals = w[np.newaxis, :] - (1.0/beta) * np.log1p(np.exp(-beta * (gaps - w[np.newaxis, :])))
    
    union_lengths = w[np.newaxis, :] + softmin_vals.sum(axis=0)
    area = np.mean(union_lengths)
    return area


def _smooth_grad(x, n=64, n_heights=20000, beta=500.0):
    """Compute gradient of smooth area approximation."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # softmin(gap, w) = w - (1/beta) * log1p(exp(-beta*(gap - w)))
    # d(softmin)/d(gap) = -exp(-beta*(gap-w)) / (1 + exp(-beta*(gap-w)))
    #                    = -sigmoid(beta*(gap - w)) ... wait
    # Let t = gap - w
    # softmin = w - (1/beta)*log(1 + exp(-beta*t))
    # d(softmin)/dt = -(1/beta) * (-beta*exp(-beta*t))/(1+exp(-beta*t)) = exp(-beta*t)/(1+exp(-beta*t)) = 1/(1+exp(beta*t))
    # = sigmoid(-beta*t)
    # So d(softmin)/d(gap) = sigmoid(-beta*(gap - w))
    
    # d(softmin)/d(w) = 1 - (1/beta)*exp(-beta*(gap-w))/(1+exp(-beta*(gap-w)))
    #                  = 1 - sigmoid(-beta*(gap-w)) = sigmoid(beta*(gap-w))
    
    # d(area)/d(x_i) = mean_y sum_j [d(softmin_j)/d(L_j) * dL_j/dx_i]
    # L_j depends on x_i only if j == i (in the unsorted sense)
    # But after sorting, the index i maps to some rank r at each height
    
    # dL_i/dx_i = 1
    # So d(area)/d(x_i) = mean_y [d(softmin_{r-1})/d(L_r) * 1 + d(softmin_r)/d(L_r) * 1]
    # where r is the rank of L_i in sorted order
    # d(softmin_{j})/d(L_{j+1}) = sigmoid(-beta*(gap_j - w))  [derivative w.r.t. the larger element]
    # d(softmin_{j})/d(L_{j}) = -sigmoid(-beta*(gap_j - w))   [derivative w.r.t. the smaller element]
    
    # Wait, let me be more careful.
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # softmin_j = w - (1/beta)*log(1 + exp(-beta*(gap_j - w)))
    # d(softmin_j)/d(L_sorted[j+1]) = d(softmin_j)/d(gap_j) = sigmoid(-beta*(gap_j - w))
    # d(softmin_j)/d(L_sorted[j]) = -d(softmin_j)/d(gap_j) = -sigmoid(-beta*(gap_j - w))
    
    # For x_i, at height m, L_i has rank r.
    # L_i appears as L_sorted[r, m].
    # It contributes to:
    # - softmin_{r-1} as L_sorted[r] = L_sorted[(r-1)+1]: derivative = sigmoid(-beta*(gap_{r-1} - w))
    # - softmin_r as L_sorted[r] = L_sorted[r]: derivative = -sigmoid(-beta*(gap_r - w))
    
    # So d(area)/d(x_i) = mean_m [sigmoid(-beta*(gap_{r-1,m} - w_m)) - sigmoid(-beta*(gap_{r,m} - w_m))]
    # with boundary conditions: if r=0, no contribution from softmin_{-1}; if r=n-1, no contribution from softmin_{n-1}
    
    sort_idx = np.argsort(L, axis=0)  # (n, n_heights)
    
    # Find rank of i at each height
    L_i = L[i, :]
    less = L < L_i[np.newaxis, :]
    rank = less.sum(axis=0)  # (n_heights,)
    
    m_idx = np.arange(n_heights)
    r = rank
    
    # Compute sigmoid terms
    # gap_{r-1,m} = L_sorted[r,m] - L_sorted[r-1,m] = L_i - L_sorted[r-1,m]
    # gap_{r,m} = L_sorted[r+1,m] - L_sorted[r,m] = L_sorted[r+1,m] - L_i
    
    L_prev = np.where(r > 0, L_sorted[np.clip(r - 1, 0, n-1), m_idx], -np.inf)
    L_next = np.where(r < n-1, L_sorted[np.clip(r + 1, 0, n-1), m_idx], np.inf)
    
    gap_prev = L_i - L_prev  # = gap_{r-1} if r > 0
    gap_next = L_next - L_i  # = gap_r if r < n-1
    
    # sigmoid(-beta * t) = 1/(1+exp(beta*t))
    # Use stable computation
    def _sigmoid_neg_beta_t(t, beta):
        # 1/(1+exp(beta*t))
        # For t > 0: exp(-beta*t)/(1+exp(-beta*t))
        # For t < 0: 1/(1+exp(beta*t))
        result = np.zeros_like(t, dtype=float)
        pos = t >= 0
        result[pos] = np.exp(-beta * t[pos]) / (1.0 + np.exp(-beta * t[pos]))
        result[~pos] = 1.0 / (1.0 + np.exp(beta * t[~pos]))
        return result
    
    sig_prev = _sigmoid_neg_beta_t(gap_prev, beta)
    sig_next = _sigmoid_neg_beta_t(gap_next, beta)
    
    # Only count valid contributions
    valid_prev = (r > 0).astype(float)
    valid_next = (r < n-1).astype(float)
    
    grad_terms = valid_prev * sig_prev - valid_next * sig_next
    
    grad_i = np.mean(grad_terms)
    return grad_i


def _smooth_grad_vec(x, n=64, n_heights=20000, beta=500.0):
    """Compute full gradient vector for smooth area."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    
    sort_idx = np.argsort(L, axis=0)  # (n, n_heights)
    
    # For each height m, for each rank r, compute the gradient contribution
    # grad_x_i = mean_m [sigmoid(-beta*(gap_{r-1,m} - w_m)) * [r>0] - sigmoid(-beta*(gap_{r,m} - w_m)) * [r<n-1]]
    
    # Build inverse rank: inv_rank[j, m] = rank of original index j at height m
    inv_rank = np.empty_like(sort_idx)
    m_idx = np.arange(n_heights)
    inv_rank[sort_idx, m_idx[np.newaxis, :]] = np.arange(n)[:, np.newaxis]
    
    # Now for each i, rank_i = inv_rank[i, :]
    # gap_prev_i = L_sorted[rank_i, m] - L_sorted[rank_i - 1, m] if rank_i > 0
    # gap_next_i = L_sorted[rank_i + 1, m] - L_sorted[rank_i, m] if rank_i < n-1
    
    r = inv_rank  # (n, n_heights)
    
    L_sorted_ext = np.concatenate([np.full((1, n_heights), -np.inf), L_sorted, np.full((1, n_heights), np.inf)], axis=0)
    
    L_prev = L_sorted_ext[r, m_idx[np.newaxis, :]]  # L_sorted[rank-1]
    L_next = L_sorted_ext[r + 2, m_idx[np.newaxis, :]]  # L_sorted[rank+1]
    
    gap_prev = L_sorted_ext[r + 1, m_idx[np.newaxis, :]] - L_prev
    gap_next = L_next - L_sorted_ext[r + 1, m_idx[np.newaxis, :]]
    
    def _sigmoid_neg_beta_t(t, beta):
        result = np.zeros_like(t, dtype=float)
        pos = t >= 0
        result[pos] = np.exp(-beta * t[pos]) / (1.0 + np.exp(-beta * t[pos]))
        result[~pos] = 1.0 / (1.0 + np.exp(beta * t[~pos]))
        return result
    
    sig_prev = _sigmoid_neg_beta_t(gap_prev, beta)
    sig_next = _sigmoid_neg_beta_t(gap_next, beta)
    
    valid_prev = (r > 0).astype(float)
    valid_next = (r < n-1).astype(float)
    
    grad_terms = valid_prev * sig_prev - valid_next * sig_next
    
    grad = np.mean(grad_terms, axis=1)
    return grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize Kakeya needle triangle positions to minimize union area.
    Uses coordinate descent, then L-BFGS-B with smooth surrogate, then final polish.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    n = 64
    x = _keich_initial(n)
    best_x = x.copy()
    best_area = _area_fast(x, n)
    
    # Phase 1: Coordinate descent with subgradient bisection
    max_passes = 15
    tol = 1e-12
    
    for pass_num in range(max_passes):
        if time.time() > deadline - 180:
            break
        
        improved = False
        
        for i in range(n):
            if time.time() > deadline - 180:
                break
            
            lo = x[i] - 0.1
            hi = x[i] + 0.1
            
            x[i] = lo
            sg_lo = _subgrad_xi(x, i, n)
            x[i] = hi
            sg_hi = _subgrad_xi(x, i, n)
            
            for _ in range(30):
                if time.time() > deadline - 180:
                    break
                
                mid = (lo + hi) / 2
                x[i] = mid
                sg_mid = _subgrad_xi(x, i, n)
                
                if sg_mid > 0:
                    hi = mid
                else:
                    lo = mid
            
            x[i] = (lo + hi) / 2
            new_area = _area_fast(x, n)
            
            if new_area < best_area - 1e-15:
                best_area = new_area
                best_x = x.copy()
                improved = True
            else:
                x[i] = best_x[i]
        
        if not improved:
            break
    
    # Phase 2: L-BFGS-B with smooth surrogate
    if time.time() < deadline - 120:
        beta = 500.0
        
        def smooth_obj(xx):
            return _smooth_area(xx, n, 20000, beta)
        
        def smooth_grad_func(xx):
            return _smooth_grad_vec(xx, n, 20000, beta)
        
        try:
            result = minimize(
                smooth_obj, best_x, method='L-BFGS-B',
                jac=smooth_grad_func,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                cand_area = _area_fast(result.x, n)
                if cand_area < best_area - 1e-15:
                    best_area = cand_area
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision coordinate descent polish
    if time.time() < deadline - 60:
        for pass_num in range(10):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                
                lo = best_x[i] - 0.01
                hi = best_x[i] + 0.01
                
                x = best_x.copy()
                x[i] = lo
                sg_lo = _subgrad_xi(x, i, n)
                x[i] = hi
                sg_hi = _subgrad_xi(x, i, n)
                
                for _ in range(40):
                    if time.time() > deadline - 30:
                        break
                    
                    mid = (lo + hi) / 2
                    x[i] = mid
                    sg_mid = _subgrad_xi(x, i, n)
                    
                    if sg_mid > 0:
                        hi = mid
                    else:
                        lo = mid
                
                x[i] = (lo + hi) / 2
                new_area = _area_fast(x, n)
                
                if new_area < best_area - 1e-15:
                    best_area = new_area
                    best_x = x.copy()
                    improved = True
        
        if not improved:
            break
    
    final_area = _area_fast(best_x, n)
    if np.isfinite(final_area) and final_area > 0:
        return best_x
    return _keich_initial(n)
>>>>>>> REPLACE
