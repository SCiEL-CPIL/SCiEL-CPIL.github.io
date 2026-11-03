<<<<<<< SEARCH
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(50, best_rank - 25), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=400,
                                    deadline=deadline - 20, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
=======
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(45, best_rank - 30), -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            
            # Alternate between random and syndrome-based initialization
            if seed % 2 == 0:
                result = _als_decompose(n, m, p, rank, seed=seed, max_iters=350,
                                        deadline=deadline - 15, polish_iters=60)
            else:
                U_syn, V_syn, W_syn = _syndrome_init(n, m, p, rank, seed=seed)
                result = _als_decompose(n, m, p, rank,
                                        U_init=U_syn, V_init=V_syn, W_init=W_syn,
                                        max_iters=350, deadline=deadline - 15, polish_iters=60)
            
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
>>>>>>> REPLACE
