<<<<<<< SEARCH
def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses multiple NEH restarts and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    best_order = None
    best_ms = float('inf')
    
    # Strategy: Multiple NEH restarts
    # Determine number of restarts based on time and N
    # If N is small, we can do more restarts.
    n_restarts = 10 if n_jobs < 30 else 5
    
    for r in range(n_restarts):
        if time.time() >= deadline - 10.0:
            break
        
        seed = r * 1000 + 42
        initial = _neh_initialization(processing_times, seed=seed)
        
        # Apply local search
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            improved, ms = _fast_local_search(initial, processing_times, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = improved
    
    if best_order is None:
        best_order = _neh_initialization(processing_times)
    
    return best_order
=======
def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses multiple NEH restarts and local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    best_order = None
    best_ms = float('inf')
    
    # Strategy: Multiple NEH restarts with adaptive time allocation
    # Start with a baseline, then use remaining time for restarts
    seed = 42
    initial = _neh_initialization(processing_times, seed=seed)
    best_order = initial
    best_ms = _makespan(best_order, processing_times)
    
    # Apply local search to the first solution
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 2.0
    if remaining > 5.0:
        improved, ms = _fast_local_search(best_order, processing_times, time_limit=remaining)
        if ms < best_ms:
            best_ms = ms
            best_order = improved
    
    # Use remaining time for additional restarts
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 1.0
    if remaining > 10.0:
        # Estimate time per restart based on first local search
        first_ls_time = min(remaining, 20.0)
        n_additional = max(1, int(remaining / max(first_ls_time, 5.0)))
        
        for r in range(1, n_additional + 1):
            if time.time() >= deadline - 5.0:
                break
            
            seed = r * 1000 + 42
            initial = _neh_initialization(processing_times, seed=seed)
            
            # Apply local search with reduced time budget
            remaining_for_ls = deadline - time.time() - 3.0
            if remaining_for_ls > 3.0:
                improved, ms = _fast_local_search(initial, processing_times, time_limit=remaining_for_ls)
                if ms < best_ms:
                    best_ms = ms
                    best_order = improved
    
    return best_order
>>>>>>> REPLACE
