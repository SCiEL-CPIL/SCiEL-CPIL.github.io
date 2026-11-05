<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    for target_rank in [49, 48, 47]:
        if _remaining() < 10:
            break
        
        params = _try_als(n, m, p, target_rank, max_restarts=40, max_iters=150)
        if params is not None:
            U, V, W = params
            rounded = _try_round_and_verify(n, m, p, U, V, W)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best_result
=======
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Theoretical lower bound for 2x5x5 is around 20-30?
    # Schoolbook is 50.
    # Let's try ranks from 49 down to 30.
    
    # First, try the new random support method for a few ranks
    for target_rank in [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if _remaining() < 15:
            break
        
        res = _try_random_support(n, m, p, target_rank, max_trials=50)
        if res is not None:
            U_r, V_r, W_r = res
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = target_rank
            break
    
    # If not found, try ALS as before (maybe with fewer restarts to save time)
    if best_rank == rank_schoolbook:
        for target_rank in [49, 48, 47]:
            if _remaining() < 10:
                break
            
            params = _try_als(n, m, p, target_rank, max_restarts=10, max_iters=50)
            if params is not None:
                U, V, W = params
                rounded = _try_round_and_verify(n, m, p, U, V, W)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    break
    
    return best_result
>>>>>>> REPLACE
