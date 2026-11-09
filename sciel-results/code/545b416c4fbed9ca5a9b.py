    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Local search with insertion moves
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline)
    
    # If we have time, try multiple restarts with different initial heuristics
    if time.time() < deadline - 10.0 and n_jobs > 5:
        # ... (long block) ...
    
    # Final polish: one more pass of local search if time remains
    if time.time() < deadline - 3.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline)
    
    return list(best_perm)
