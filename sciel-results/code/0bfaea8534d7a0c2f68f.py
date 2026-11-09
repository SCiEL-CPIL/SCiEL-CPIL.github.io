<<<<<<< SEARCH
    # Local search with 2-opt
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_iters=200, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try swap local search
    if time.time() < deadline:
        swp_perm = _local_search_swap(best_perm, pt, max_iters=150, deadline=deadline)
        swp_ms = _makespan_fast(swp_perm, pt)
        if swp_ms < best_ms:
            best_perm = swp_perm
            best_ms = swp_ms
    
    # Try insertion local search
    if time.time() < deadline:
        ins_perm = _local_search_insert(best_perm, pt, max_iters=80, deadline=deadline)
        ins_ms = _makespan_fast(ins_perm, pt)
        if ins_ms < best_ms:
            best_perm = ins_perm
            best_ms = ins_ms
    
    # Tabu search to escape local optima
    if time.time() < deadline:
        tabu_perm = _tabu_search(best_perm, pt, deadline, tabu_tenure=15, max_iters=3000)
        tabu_ms = _makespan_fast(tabu_perm, pt)
        if tabu_ms < best_ms:
            best_perm = tabu_perm
            best_ms = tabu_ms
    
    # Simulated annealing to escape local optima
    if time.time() < deadline:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, n_restarts=5)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 8
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick local search from random start
            ls_r = _local_search_swap(random_perm, pt, max_iters=50, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # If improved, try insertion local search
            if time.time() < deadline:
                ls_r2 = _local_search_insert(ls_r, pt, max_iters=30, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
    
    # Final polish with swap local search
    if time.time() < deadline:
        final_perm = _local_search_swap(best_perm, pt, max_iters=100, deadline=deadline)
        final_ms = _makespan_fast(final_perm, pt)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    # Final polish with insertion local search
    if time.time() < deadline:
        final_perm2 = _local_search_insert(best_perm, pt, max_iters=50, deadline=deadline)
        final_ms2 = _makespan_fast(final_perm2, pt)
        if final_ms2 < best_ms:
            best_perm = final_perm2
            best_ms = final_ms2
    
    return list(best_perm)
=======
    # Local search with 2-opt
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_iters=300, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try swap local search
    if time.time() < deadline:
        swp_perm = _local_search_swap(best_perm, pt, max_iters=250, deadline=deadline)
        swp_ms = _makespan_fast(swp_perm, pt)
        if swp_ms < best_ms:
            best_perm = swp_perm
            best_ms = swp_ms
    
    # Try insertion local search
    if time.time() < deadline:
        ins_perm = _local_search_insert(best_perm, pt, max_iters=120, deadline=deadline)
        ins_ms = _makespan_fast(ins_perm, pt)
        if ins_ms < best_ms:
            best_perm = ins_perm
            best_ms = ins_ms
    
    # Tabu search to escape local optima
    if time.time() < deadline:
        tabu_perm = _tabu_search(best_perm, pt, deadline, tabu_tenure=12, max_iters=10000)
        tabu_ms = _makespan_fast(tabu_perm, pt)
        if tabu_ms < best_ms:
            best_perm = tabu_perm
            best_ms = tabu_ms
    
    # Simulated annealing to escape local optima
    if time.time() < deadline:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, n_restarts=8)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 12
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick local search from random start
            ls_r = _local_search_swap(random_perm, pt, max_iters=80, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # If improved, try insertion local search
            if time.time() < deadline:
                ls_r2 = _local_search_insert(ls_r, pt, max_iters=50, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
    
    # Final polish with swap local search
    if time.time() < deadline:
        final_perm = _local_search_swap(best_perm, pt, max_iters=150, deadline=deadline)
        final_ms = _makespan_fast(final_perm, pt)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    # Final polish with insertion local search
    if time.time() < deadline:
        final_perm2 = _local_search_insert(best_perm, pt, max_iters=80, deadline=deadline)
        final_ms2 = _makespan_fast(final_perm2, pt)
        if final_ms2 < best_ms:
            best_perm = final_perm2
            best_ms = final_ms2
    
    return list(best_perm)
>>>>>>> REPLACE
