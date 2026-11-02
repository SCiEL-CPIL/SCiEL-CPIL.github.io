    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), current_rank
    best_solution = None
    
    for target_rank in range(45, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        prev_sol = best_solution if best_solution is not None else None
        result = try_als_l_bfgs(target_rank, num_trials=3, max_iter_als=10, max_iter_lbfgs=200, prev_solution=prev_sol)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            best_solution = (U_t, V_t, W_t)
            # Do not break, continue to try lower ranks
