<<<<<<< SEARCH
def _pairwise_swap_search(heights, time_budget=10, start_time=0):
    """
    Pairwise swap local search: try swapping signs at two positions to reduce C3 bound.
    This explores a larger neighborhood than single flips.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    while improved:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        best_gain = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of pairs to keep computation manageable
        # Use random sampling of pairs
        np.random.seed(42)
        indices = np.random.choice(N, size=min(50, N), replace=False)
        
        for ii in range(len(indices)):
            i = indices[ii]
            for jj in range(ii + 1, len(indices)):
                j = indices[jj]
                if time.time() - start_time > time_budget:
                    break
                
                # Swap signs
                h[i], h[j] = -h[i], -h[j]
                new_bound = _compute_c3_bound(h)
                h[i], h[j] = -h[i], -h[j]
                
                gain = current_bound - new_bound
                if gain > best_gain:
                    best_gain = gain
                    best_i, best_j = i, j
        
        if best_i >= 0:
            h[best_i], h[best_j] = -h[best_i], -h[best_j]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound
=======
def _pairwise_swap_search(heights, time_budget=10, start_time=0):
    """
    Pairwise swap local search: try swapping signs at two positions to reduce C3 bound.
    Uses incremental convolution updates for speed.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    conv = _compute_convolution(h)
    current_max = np.max(np.abs(conv))
    current_bound = dx * current_max
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    while improved:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        best_gain = 0
        best_i, best_j = -1, -1
        best_new_conv = None
        best_new_max = current_max
        
        # Sample a subset of pairs to keep computation manageable
        np.random.seed(42 + iters if 'iters' in dir() else 42)
        indices = np.random.choice(n, size=min(40, n), replace=False)
        
        for ii in range(len(indices)):
            i = indices[ii]
            for jj in range(ii + 1, len(indices)):
                j = indices[jj]
                if time.time() - start_time > time_budget:
                    break
                
                # Flipping both i and j
                # Delta is sum of individual deltas
                old_hi = h[i]
                old_hj = h[j]
                
                # Compute combined delta
                lags = np.arange(-(n-1), n)
                
                # Delta from flipping i
                i_plus = i + lags
                i_minus = i - lags
                val1_i = np.where((i_plus >= 0) & (i_plus < n), 
                                  h[np.clip(i_plus, 0, n-1)], 0.0)
                val2_i = np.where((i_minus >= 0) & (i_minus < n), 
                                  h[np.clip(i_minus, 0, n-1)], 0.0)
                delta_i = -2 * old_hi * (val1_i + val2_i)
                
                # Delta from flipping j
                j_plus = j + lags
                j_minus = j - lags
                val1_j = np.where((j_plus >= 0) & (j_plus < n), 
                                  h[np.clip(j_plus, 0, n-1)], 0.0)
                val2_j = np.where((j_minus >= 0) & (j_minus < n), 
                                  h[np.clip(j_minus, 0, n-1)], 0.0)
                delta_j = -2 * old_hj * (val1_j + val2_j)
                
                # Combined: after flipping i, the delta from flipping j changes
                # because h[i] has changed. Let's be more careful.
                # Actually, flipping i changes h[i] to -h[i]. Then flipping j changes h[j] to -h[j].
                # The net effect on conv is more complex. Let's just do it directly for pairs.
                
                # Simpler: temporarily flip both, compute new conv, restore
                h[i], h[j] = -h[i], -h[j]
                new_conv = _compute_convolution(h)
                h[i], h[j] = -h[i], -h[j]
                
                new_max = np.max(np.abs(new_conv))
                gain = current_max - new_max
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_i, best_j = i, j
                    best_new_conv = new_conv
                    best_new_max = new_max
        
        if best_i >= 0:
            h[best_i], h[best_j] = -h[best_i], -h[best_j]
            conv = best_new_conv
            current_max = best_new_max
            current_bound = dx * current_max
            improved = True
    
    return h, current_bound
>>>>>>> REPLACE

