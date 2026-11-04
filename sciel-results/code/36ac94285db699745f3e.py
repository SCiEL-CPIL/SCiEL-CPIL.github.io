def _try_rank_one(n, m, p):
    """
    Try to find a rank 1 decomposition.
    If the target tensor is the all-ones tensor, U=V=W=1 works.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 1 first
    result = _try_rank_one(n, m, p)
    if result is not None:
        return result
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
