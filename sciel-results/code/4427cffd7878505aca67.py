<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill
    
    # Schoolbook baseline
    rank0 = n * m * p
    U0 = [[0]*rank0 for _ in range(n*m)]
    V0 = [[0]*rank0 for _ in range(m*p)]
    W0 = [[0]*rank0 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Lower bound: max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound up to rank0-1
    ranks_to_try = list(range(lower_bound, rank0))
    
    # Prioritize ranks near the lower bound
    ranks_to_try.sort(key=lambda r: abs(r - lower_bound))
    
    for r in ranks_to_try:
        if time.time() > deadline - 3:
            break
        
        # Adaptive number of restarts based on rank
        if r <= lower_bound + 3:
            n_restarts = 20
        elif r <= lower_bound + 10:
            n_restarts = 10
        else:
            n_restarts = 5
        
        result = _try_rank_search(n, m, p, r, deadline, n_restarts=n_restarts)
        
        if result is not None:
            U_r, V_r, W_r = result
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
    
    return {"U": best_U, "V": best_V, "W": best_W}
=======
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill
    
    # Schoolbook baseline
    rank0 = n * m * p
    U0 = [[0]*rank0 for _ in range(n*m)]
    V0 = [[0]*rank0 for _ in range(m*p)]
    W0 = [[0]*rank0 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Lower bound: max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound up to rank0-1
    ranks_to_try = list(range(lower_bound, rank0))
    
    # Prioritize ranks near the lower bound
    ranks_to_try.sort(key=lambda r: abs(r - lower_bound))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        # Adaptive number of restarts based on rank
        # Give more time to the most promising ranks
        if r <= lower_bound + 2:
            n_restarts = 30
        elif r <= lower_bound + 5:
            n_restarts = 15
        else:
            n_restarts = 5
        
        result = _try_rank_search(n, m, p, r, deadline, n_restarts=n_restarts)
        
        if result is not None:
            U_r, V_r, W_r = result
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                # If we found a solution near the bound, we might be done
                if r <= lower_bound + 5:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
>>>>>>> REPLACE
