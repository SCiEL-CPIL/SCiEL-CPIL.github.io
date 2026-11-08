    # Try rank 25 first (remove 1 column)
    if time.time() < deadline - 10:
        result = try_reduce(25, 1, max_combos=26, local_iters_per_combo=5000)
        if result:
            U_sol, V_sol, W_sol = result
            if check(U_sol, V_sol, W_sol):
                best_rank = 25
                best_U, best_V, best_W = U_sol, V_sol, W_sol
    
    # Try rank 24 (remove 2 columns)
    if time.time() < deadline - 10 and best_rank == 26:
        result = try_reduce(24, 2, max_combos=100, local_iters_per_combo=2000)
        if result:
            U_sol, V_sol, W_sol = result
            if check(U_sol, V_sol, W_sol):
                best_rank = 24
                best_U, best_V, best_W = U_sol, V_sol, W_sol
