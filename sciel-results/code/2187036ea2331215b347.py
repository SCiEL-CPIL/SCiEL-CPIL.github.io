def _try_structured_simple(n, m, p):
    """
    Try simple structured decompositions:
    1. U is Identity (Rank nm)
    2. V is Identity (Rank mp)
    3. W is Identity (Rank pn)
    Returns the best (lowest rank) valid decomposition found.
    """
    best = None
    best_rank = float('inf')
    
    # 1. U is Identity
    rank = n * m
    U = np.eye(rank)
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        if rank < best_rank:
            best_rank = rank
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 2. V is Identity
    rank = m * p
    U = np.ones((n * m, rank))
    V = np.eye(rank)
    W = np.ones((p * n, rank))
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        if rank < best_rank:
            best_rank = rank
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 3. W is Identity
    rank = p * n
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.eye(rank)
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        if rank < best_rank:
            best_rank = rank
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return best

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Try simple structured solutions first
    simple_sol = _try_structured_simple(n, m, p)
    if simple_sol:
        if len(simple_sol["U"][0]) < best_rank:
            best = simple_sol
            best_rank = len(simple_sol["U"][0])
            
    # Search for even lower ranks
    # Lower bound is at least max(n, m, p)
    lower_bound = max(n, m, p)
    
    # We only search if we have time and haven't reached a very low rank
    for rank in range(lower_bound, best_rank):
        if _remaining() < 15.0:
            break
        
        found = False
        
        # Try low-rank exact with multiple initializations
        for attempt in range(3):
            if _remaining() < 8.0:
                break
            
            result = _try_low_rank_exact(n, m, p, rank, max_iter=100, seed=rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
            
    # Final check
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
        
    return best
