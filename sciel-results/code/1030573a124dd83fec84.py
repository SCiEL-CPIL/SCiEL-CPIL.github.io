<<<<<<< SEARCH
def _optimize_aspect_ratio(x, y, r, w, h, time_budget=8.0):
    """Try to improve the aspect ratio by adjusting w and re-optimizing."""
    t0 = time.time()
    best = (x.copy(), y.copy(), r.copy(), w, h)
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return best
    
    # Try different aspect ratios
    for w_new in [w * 0.9, w * 1.1, w - 0.1, w + 0.1]:
        if time.time() - t0 > time_budget:
            break
        h_new = 2.0 - w_new
        if h_new < 0.3 or w_new < 0.3:
            continue
        
        # Scale positions
        scale_x = w_new / w
        scale_y = h_new / h
        x_new = x * scale_x
        y_new = y * scale_y
        
        # Try to expand radii
        r_new = _expand_radii(x_new, y_new, r, w_new, h_new)
        val = _sum_radii(x_new, y_new, r_new, w_new, h_new)
        if val > best_val:
            best_val = val
            best = (x_new.copy(), y_new.copy(), r_new.copy(), w_new, h_new)
    
    return best
=======
def _optimize_aspect_ratio(x, y, r, w, h, time_budget=10.0):
    """Try to improve the aspect ratio by adjusting w and re-optimizing with SA."""
    t0 = time.time()
    best = (x.copy(), y.copy(), r.copy(), w, h)
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return best
    
    # Try different aspect ratios with finer granularity
    w_candidates = np.linspace(0.5, 1.5, 11)  # w from 0.5 to 1.5, h = 2-w
    for w_new in w_candidates:
        if time.time() - t0 > time_budget:
            break
        h_new = 2.0 - w_new
        if h_new < 0.4 or w_new < 0.4:
            continue
        if abs(w_new - w) < 0.01:
            continue
        
        # Scale positions proportionally
        scale_x = w_new / w
        scale_y = h_new / h
        x_new = x * scale_x
        y_new = y * scale_y
        
        # Expand radii
        r_new = _expand_radii(x_new, y_new, r, w_new, h_new)
        
        # Quick local polish
        x_pol, y_pol, r_pol = _local_search_polish(x_new, y_new, r_new, w_new, h_new, 
                                                    time_budget=min(2.0, time_budget - (time.time() - t0)))
        
        val = _sum_radii(x_pol, y_pol, r_pol, w_new, h_new)
        if val > best_val:
            best_val = val
            best = (x_pol.copy(), y_pol.copy(), r_pol.copy(), w_new, h_new)
    
    return best
>>>>>>> REPLACE

