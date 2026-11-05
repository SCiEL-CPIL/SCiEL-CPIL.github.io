<<<<<<< SEARCH
def _local_search(S, sums, deadline, rng, max_iters=2000):
    """
    Local search with:
      - Try to add a random element
      - If fails, try 1-remove-1-add (swap)
      - Occasionally try 1-remove-2-add (net +1)
    """
    S = list(S)
    sums = set(sums)
    in_S = set(S)
    best_size = len(S)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random element
        candidates = [x for x in range(1, N_MAX + 1) if x not in in_S]
        if not candidates:
            break
        
        x_add = rng.choice(candidates)
        if _check_conflict(x_add, S, sums):
            _add_element(x_add, S, sums)
            in_S.add(x_add)
            best_size = len(S)
            continue
        
        # Try swap: remove random element, add x_add
        if len(S) > 1:
            idx_rm = rng.randint(0, len(S) - 1)
            x_rm = S[idx_rm]
            
            _remove_element(idx_rm, S, sums)
            in_S.discard(x_rm)
            
            if _check_conflict(x_add, S, sums):
                _add_element(x_add, S, sums)
                in_S.add(x_add)
            else:
                # Restore
                _add_element(x_rm, S, sums)
                in_S.add(x_rm)
        
        # Occasionally try net +1: remove 1, add 2
        if it % 20 == 0 and len(S) > 2:
            idx_rm = rng.randint(0, len(S) - 1)
            x_rm = S[idx_rm]
            
            _remove_element(idx_rm, S, sums)
            in_S.discard(x_rm)
            
            # Try to add two elements
            added = 0
            for _ in range(30):
                if time.time() > deadline:
                    break
                cands = [x for x in range(1, N_MAX + 1) if x not in in_S]
                if not cands:
                    break
                xa = rng.choice(cands)
                if _check_conflict(xa, S, sums):
                    _add_element(xa, S, sums)
                    in_S.add(xa)
                    added += 1
                    if added == 2:
                        break
            
            if added == 0:
                # Restore
                _add_element(x_rm, S, sums)
                in_S.add(x_rm)
    
    return S, sums
=======
def _local_search(S, sums, deadline, rng, max_iters=2000):
    """
    Local search with:
      - Try to add a random element
      - If fails, try 1-remove-1-add (swap)
      - Occasionally try 1-remove-2-add (net +1)
    """
    S = list(S)
    sums = set(sums)
    in_S = set(S)
    best_size = len(S)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random element
        candidates = [x for x in range(1, N_MAX + 1) if x not in in_S]
        if not candidates:
            break
        
        x_add = rng.choice(candidates)
        if _check_conflict(x_add, S, sums):
            _add_element(x_add, S, sums)
            in_S.add(x_add)
            best_size = len(S)
            continue
        
        # Try swap: remove random element, add x_add
        if len(S) > 1:
            idx_rm = rng.randint(0, len(S) - 1)
            x_rm = S[idx_rm]
            
            _remove_element(idx_rm, S, sums)
            in_S.discard(x_rm)
            
            if _check_conflict(x_add, S, sums):
                _add_element(x_add, S, sums)
                in_S.add(x_add)
            else:
                _add_element(x_rm, S, sums)
                in_S.add(x_rm)
        
        # Occasionally try net +1: remove 1, add 2
        if it % 15 == 0 and len(S) > 2:
            idx_rm = rng.randint(0, len(S) - 1)
            x_rm = S[idx_rm]
            
            _remove_element(idx_rm, S, sums)
            in_S.discard(x_rm)
            
            added = 0
            for _ in range(40):
                if time.time() > deadline:
                    break
                cands = [x for x in range(1, N_MAX + 1) if x not in in_S]
                if not cands:
                    break
                xa = rng.choice(cands)
                if _check_conflict(xa, S, sums):
                    _add_element(xa, S, sums)
                    in_S.add(xa)
                    added += 1
                    if added == 2:
                        break
            
            if added == 0:
                _add_element(x_rm, S, sums)
                in_S.add(x_rm)
    
    return S, sums
>>>>>>> REPLACE

