<<<<<<< SEARCH
def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Initialize with spectral method
    p = _spectral_init(a, b)
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # ILS loop
    while time.time() - start_time < deadline:
        # Local search to convergence
        p, current_cost = _local_search_steepest(a, b, p, start_time, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time
        if time.time() - start_time > deadline:
            break
        
        # Perturbation: random swaps
        # Scale number of swaps with n
        num_swaps = max(2, n // 10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            p[i], p[j] = p[j], p[i]
            # Update cost approximately (will be recomputed in local search)
            # For now, just track; local search will find the true cost
            # Actually, we need to update current_cost for the next local search
            # Let's recompute cost after perturbation
            current_cost = _qap_cost(a, b, p)
        
        # If we've used too much time, break
        if time.time() - start_time > deadline:
            break
    
    return best_p.tolist()
=======
def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Initialize with spectral method
    p = _spectral_init(a, b)
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # Track consecutive non-improvements for adaptive perturbation
    consecutive_no_improve = 0
    
    # ILS loop
    while time.time() - start_time < deadline:
        # Local search to convergence, tracking best single-swap delta
        p, current_cost, best_swap_delta = _local_search_steepest(a, b, p, start_time, deadline, track_best=True)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        # Check time
        if time.time() - start_time > deadline:
            break
        
        # Adaptive perturbation: more swaps when stuck, fewer when improving
        base_swaps = max(2, n // 10)
        if consecutive_no_improve >= 5:
            num_swaps = base_swaps * 2
        elif consecutive_no_improve >= 2:
            num_swaps = base_swaps + 1
        else:
            num_swaps = base_swaps
        
        # Batch all swaps, then compute cost once
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            p[i], p[j] = p[j], p[i]
        
        current_cost = _qap_cost(a, b, p)
        
        # If we've used too much time, break
        if time.time() - start_time > deadline:
            break
    
    # Final polish: run local search multiple times from best_p
    # Use remaining time for a thorough polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        polish_deadline = start_time + deadline - 2.0  # 2s safety margin
        polish_p = best_p.copy()
        polish_cost = best_cost
        # Multiple polish passes with fresh local search
        for _ in range(3):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost = _local_search_steepest(a, b, polish_p, start_time, polish_deadline)
            if polish_cost < best_cost:
                best_cost = polish_cost
                best_p = polish_p.copy()
            # Small perturbation between polish passes
            if time.time() > polish_deadline:
                break
            for _ in range(2):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    polish_p[i], polish_p[j] = polish_p[j], polish_p[i]
            polish_cost = _qap_cost(a, b, polish_p)
    
    return best_p.tolist()
>>>>>>> REPLACE
