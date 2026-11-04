<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward
    # We limit the search to a reasonable range to stay within time limits
    max_rank_to_try = min(35, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds for each rank
        max_seeds = 10 if rank < 30 else 5
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            
            result = _als_decompose(n, m, p, rank, seed, max_iter=50)
            if result is not None:
                U_opt, V_opt, W_opt, err = result
                # Round to nearest 0.5 if close
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
    
    return best_result
=======
def _polish_decomposition(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """
    High-precision polish pass: alternate between fixing two matrices and 
    solving for the third, with tighter convergence.
    """
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]
    
    for it in range(max_iter):
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                A = V[j*p:(j+1)*p, :] * W[i::n, :]
                b = np.ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m + j, :] = u
                except:
                    pass
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A = U[:, j::m] * W[k::n, :]
                b = np.ones(n)
                try:
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p + k, :] = v
                except:
                    pass
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A = U[i*m:(i+1)*m, :] * V[k::p, :]
                b = np.ones(m)
                try:
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n + i, :] = w
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err += (val - 1.0) ** 2
        
        if err < tol:
            break
    
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward
    max_rank_to_try = min(35, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        # Adaptive seed count based on remaining time
        remaining = deadline - 15 - time.time()
        if rank == min_rank:
            max_seeds = min(15, max(3, int(remaining / 5)))
        else:
            max_seeds = min(8, max(2, int(remaining / 8)))
        
        found = False
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            
            result = _als_decompose(n, m, p, rank, seed, max_iter=60)
            if result is not None:
                U_opt, V_opt, W_opt, err = result
                
                # Try rounding to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
                
                # If rounding to 0.5 didn't work, try polishing and rounding
                if time.time() < deadline - 20:
                    U_pol, V_pol, W_pol, pol_err = _polish_decomposition(
                        U_opt, V_opt, W_opt, n, m, p, max_iter=100, tol=1e-12)
                    
                    U_pol_round = np.round(U_pol * 2) / 2
                    V_pol_round = np.round(V_pol * 2) / 2
                    W_pol_round = np.round(W_pol * 2) / 2
                    
                    U_pol_list = U_pol_round.tolist()
                    V_pol_list = V_pol_round.tolist()
                    W_pol_list = W_pol_round.tolist()
                    
                    if _verify(U_pol_list, V_pol_list, W_pol_list, n, m, p):
                        best_result = {"U": U_pol_list, "V": V_pol_list, "W": W_pol_list}
                        best_rank = rank
                        return best_result
    
    return best_result
>>>>>>> REPLACE
