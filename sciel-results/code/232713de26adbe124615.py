<<<<<<< SEARCH
def _project_symmetric(params, target_sum):
    """Project symmetric params to satisfy sum(h) = target_sum and 0<=h<=1."""
    n_half = len(params)
    n = 2 * n_half
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    
    # Target sum for h is n/2
    # Sum(h) = 2 * sum(params)
    # So we need sum(params) = n/4
    target_param_sum = n / 4.0
    
    # Project params to sum = target_param_sum, bounds [0,1]
    p = params.copy()
    # Simple projection: clip and adjust
    p = np.clip(p, 0, 1)
    diff = target_param_sum - np.sum(p)
    if abs(diff) > 1e-12:
        # Distribute diff to elements not at bounds
        # For simplicity, distribute evenly and clip
        p += diff / n_half
        p = np.clip(p, 0, 1)
        # Iterate a few times for better accuracy
        for _ in range(5):
            diff = target_param_sum - np.sum(p)
            if abs(diff) < 1e-12: break
            mask = (p > 0) & (p < 1)
            if np.sum(mask) == 0:
                # If all at bounds, we can't adjust. 
                # But target is n/4, max sum is n/2, min 0. 
                # n/4 is in range.
                break
            p[mask] += diff / np.sum(mask)
            p = np.clip(p, 0, 1)
    return p
=======
def _project_symmetric(params, target_sum):
    """Project symmetric params to satisfy sum(h) = target_sum and 0<=h<=1."""
    n_half = len(params)
    n = 2 * n_half
    
    # Target sum for h is n/2
    # Sum(h) = 2 * sum(params)
    # So we need sum(params) = n/4
    target_param_sum = n / 4.0
    
    # Project params to sum = target_param_sum, bounds [0,1]
    p = params.copy()
    p = np.clip(p, 0, 1)
    diff = target_param_sum - np.sum(p)
    if abs(diff) > 1e-14:
        for _ in range(50):
            diff = target_param_sum - np.sum(p)
            if abs(diff) < 1e-14: break
            mask = (p > 1e-12) & (p < 1 - 1e-12)
            if np.sum(mask) == 0:
                # Try with looser bounds
                mask = (p > 0) & (p < 1)
                if np.sum(mask) == 0:
                    break
            p[mask] += diff / np.sum(mask)
            p = np.clip(p, 0, 1)
    return p
>>>>>>> REPLACE

