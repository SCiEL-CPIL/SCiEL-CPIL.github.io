<<<<<<< SEARCH
def _multiple_restart_search(deadline):
    """Multiple restarts with different initial strategies."""
    best = [1, 2, 5, 11]
    
    # Strategy 1: Simple greedy
    a1 = [1, 2, 5, 11]
    sums1 = _compute_sums(a1)
    for x in range(12, 285):
        if time.time() > deadline:
            break
        if _can_add(x, a1, sums1):
            _add_element(x, a1, sums1)
    if len(a1) > len(best):
        best = list(a1)
    
    # Strategy 2: Greedy with backtracking
    a2 = _greedy_with_backtrack([1, 2, 5, 11], deadline, 20)
    if len(a2) > len(best):
        best = list(a2)
    
    # Strategy 3: Randomized restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start with prescribed elements plus random additions
        a3 = [1, 2, 5, 11]
        sums3 = _compute_sums(a3)
        
        # Add some random elements first
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        for x in candidates[:10]:
            if _can_add(x, a3, sums3):
                _add_element(x, a3, sums3)
        
        # Then greedy fill
        for x in range(12, 285):
            if time.time() > deadline:
                break
            if x in a3:
                continue
            if _can_add(x, a3, sums3):
                _add_element(x, a3, sums3)
        
        if len(a3) > len(best):
            best = list(a3)
        
        # Try backtracking from this state
        if len(a3) >= len(best):
            current_set = set(a3)
            candidates_bt = [x for x in range(1, 285) if x not in current_set]
            best_bt = list(a3)
            _backtrack_search(a3, sums3, candidates_bt, 0, 4, deadline, best_bt)
            if len(best_bt) > len(best):
                best = list(best_bt)
    
    return best
=======
def _multiple_restart_search(deadline):
    """Multiple restarts with different initial strategies."""
    best = [1, 2, 5, 11]
    
    # Strategy 1: Simple greedy + local search
    a1 = [1, 2, 5, 11]
    sums1 = _compute_sums(a1)
    for x in range(12, 285):
        if time.time() > deadline:
            break
        if _can_add(x, a1, sums1):
            _add_element(x, a1, sums1)
    a1, sums1 = _local_search(a1, sums1, deadline)
    if len(a1) > len(best):
        best = list(a1)
    
    # Strategy 2: Greedy with backtracking
    a2 = _greedy_with_backtrack([1, 2, 5, 11], deadline, 20)
    if len(a2) > len(best):
        best = list(a2)
    
    # Strategy 3: Randomized restarts with local search
    for restart in range(30):
        if time.time() > deadline:
            break
        
        a3 = [1, 2, 5, 11]
        sums3 = _compute_sums(a3)
        
        # Add some random elements first
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        for x in candidates[:15]:
            if _can_add(x, a3, sums3):
                _add_element(x, a3, sums3)
        
        # Then greedy fill
        for x in range(12, 285):
            if time.time() > deadline:
                break
            if x in a3:
                continue
            if _can_add(x, a3, sums3):
                _add_element(x, a3, sums3)
        
        # Local search polish
        a3, sums3 = _local_search(a3, sums3, deadline)
        
        if len(a3) > len(best):
            best = list(a3)
        
        # Try backtracking from this state if promising
        if len(a3) >= len(best) and time.time() < deadline:
            current_set = set(a3)
            candidates_bt = [x for x in range(1, 285) if x not in current_set]
            best_bt = list(a3)
            _backtrack_search(a3, sums3, candidates_bt, 0, 4, deadline, best_bt)
            if len(best_bt) > len(best):
                best = list(best_bt)
    
    return best
>>>>>>> REPLACE

