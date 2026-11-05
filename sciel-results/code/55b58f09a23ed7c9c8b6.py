<<<<<<< SEARCH
    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 8.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 8.0, max_iter=200000, seed=7, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish with insertion local search
    if time.time() < deadline - 2.0:
        perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 1.0, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final 2-opt polish
    if time.time() < deadline - 0.5:
        perm, ms = _local_search_2opt(pt_list, best_perm, deadline - 0.5, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
=======
    # Tabu search from best to escape local optima with block moves
    if time.time() < deadline - 10.0:
        perm, ms = _tabu_search(pt_list, best_perm, deadline - 10.0, max_iter=30000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish with insertion local search
    if time.time() < deadline - 2.0:
        perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 1.0, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final 2-opt polish
    if time.time() < deadline - 0.5:
        perm, ms = _local_search_2opt(pt_list, best_perm, deadline - 0.5, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
>>>>>>> REPLACE
