<<<<<<< SEARCH
    # 4. Deep Polish on top candidates
    # Sort candidates by value descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3 distinct candidates to avoid redundant work
    polished_candidates = []
    seen_keys = set()
    for val, circ, ww, hh in candidates:
        # Simple key: rounded dimensions and value
        key = (round(ww, 2), round(hh, 2), round(val, 4))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        polished_candidates.append((val, circ, ww, hh))
        if len(polished_candidates) >= 3:
            break

    for val, circ, ww, hh in polished_candidates:
        if time.time() > deadline - 10.0:
            break
        
        try:
            polished = _slsqp_polish(circ, ww, hh, deadline)
            if polished is not None:
                new_val = _total_radius(polished)
                if new_val > best_value:
                    best_value = new_val
                    best_result = polished.copy()
        except Exception:
            pass
=======
    # 4. Deep Polish on top candidates
    # Sort candidates by value descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3 distinct candidates to avoid redundant work
    polished_candidates = []
    seen_keys = set()
    for val, circ, ww, hh in candidates:
        # Simple key: rounded dimensions and value
        key = (round(ww, 2), round(hh, 2), round(val, 4))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        polished_candidates.append((val, circ, ww, hh))
        if len(polished_candidates) >= 3:
            break

    for val, circ, ww, hh in polished_candidates:
        if time.time() > deadline - 10.0:
            break
        
        try:
            polished = _slsqp_polish(circ, ww, hh, deadline)
            if polished is not None:
                new_val = _total_radius(polished)
                if new_val > best_value:
                    best_value = new_val
                    best_result = polished.copy()
        except Exception:
            pass
>>>>>>> REPLACE

