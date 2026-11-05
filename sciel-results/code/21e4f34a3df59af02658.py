<<<<<<< SEARCH
    w_urgency_vals = [0.5, 1.0, 1.5, 2.0]
    w_balance_vals = [0.5, 1.0, 1.5, 2.0]
    
    best_val = float('inf')
    best_p = (1.0, 1.0, 0)
    
    for w_u in w_urgency_vals:
        for w_b in w_balance_vals:
            p = (w_u, w_b, 0)
            v = _cache_params(p)
            if v < best_val:
                best_val = v
                best_p = p
            if _time.time() - _dispatch_start_time > _DEADLINE - _SAFETY_MARGIN:
                _best_params = best_p
                return
    _best_params = best_p
=======
    w_urgency_vals = [0.5, 1.0, 1.5, 2.0]
    w_balance_vals = [0.5, 1.0, 1.5, 2.0]
    
    best_val = float('inf')
    best_p = (1.0, 1.0, 0)
    
    for w_u in w_urgency_vals:
        for w_b in w_balance_vals:
            p = (w_u, w_b, 0)
            v = _cache_params(p)
            if v < best_val:
                best_val = v
                best_p = p
            if _time.time() - _dispatch_start_time > _DEADLINE - _SAFETY_MARGIN:
                _best_params = best_p
                return
    _best_params = best_p
>>>>>>> REPLACE
