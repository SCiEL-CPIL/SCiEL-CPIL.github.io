<<<<<<< SEARCH
    # Evaluate all initial solutions
    best_perm = None
    best_ms = float('inf')
    comp_cache = np.zeros(m, dtype=float)
    
    for perm in initial_solutions:
        ms = _makespan_fast(perm, processing_times, comp_cache)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
=======
    # Evaluate all initial solutions
    best_perm = None
    best_ms = float('inf')
    
    for perm in initial_solutions:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
>>>>>>> REPLACE

