def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try ranks
    # Known lower bound is ~18, upper bound 24.
    # Try 21, 22, 23, 24 first?
    # Or just sweep 20 to 30.
    
    for rank in range(20, 31):
        if time.time() > deadline - 10:
            break
        
        # More restarts for lower ranks
        num_restarts = 20 if rank < 24 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Deterministic seed for reproducibility
            seed = hash((rank, restart, n, m, p)) & 0xFFFFFFFF
            np.random.seed(seed)
            
            # Initialization
            # Try random
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            # Refine
            U_ref, V_ref, W_ref, err = _refine(U_init, V_init, W_init, n, m, p, max_iter=200)
            
            if err < 1e-8:
                # Round to half-integers
                U_round = _round_to_half(U_ref)
                V_round = _round_to_half(V_ref)
                W_round = _round_to_half(W_ref)
                
                valid, _ = _verify(U_round, V_round, W_round, n, m, p)
                if valid:
                    best_rank = rank
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return best_result
