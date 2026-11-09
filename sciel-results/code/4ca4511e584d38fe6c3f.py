    time_left = deadline - time.time()
    
    # Local Search first to tighten
    if time_left > 5:
        ls_rows, ls_scope = _local_search(best_rows, n_rows, k, min(5.0, time_left * 0.2))
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
            
    # SA for rest of time
    time_left = deadline - time.time()
    if time_left > 5:
        sa_rows, sa_scope = _sa_search(best_rows, n_rows, k, time_left - 2.0)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
