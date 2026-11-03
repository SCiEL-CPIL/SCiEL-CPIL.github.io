<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try both baselines and pick the best (lowest rank)
    U_base1, V_base1, W_base1 = _strassen_construct(n, m, p)
    rank1 = len(U_base1[0])
    
    U_base2, V_base2, W_base2 = _kron_strassen_construct(n, m, p)
    rank2 = len(U_base2[0])
    
    if rank1 <= rank2:
        U_base, V_base, W_base = U_base1, V_base1, W_base1
        best_rank = rank1
    else:
        U_base, V_base, W_base = U_base2, V_base2, W_base2
        best_rank = rank2
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    
    # Try to reduce rank from the best baseline
    if time.time() < deadline - 40:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 40)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(40, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(12):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=250, deadline=deadline - 20)
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
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try both baselines and pick the best (lowest rank)
    U_base1, V_base1, W_base1 = _strassen_construct(n, m, p)
    rank1 = len(U_base1[0])
    
    U_base2, V_base2, W_base2 = _kron_strassen_construct(n, m, p)
    rank2 = len(U_base2[0])
    
    if rank1 <= rank2:
        U_base, V_base, W_base = U_base1, V_base1, W_base1
        best_rank = rank1
    else:
        U_base, V_base, W_base = U_base2, V_base2, W_base2
        best_rank = rank2
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    
    # Try to reduce rank from the best baseline
    if time.time() < deadline - 40:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 40)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random initializations at various ranks with more attempts
    for rank in range(best_rank - 1, max(40, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=300, deadline=deadline - 20, polish_iters=300)
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
