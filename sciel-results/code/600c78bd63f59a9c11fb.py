<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank-51 Strassen-based decomposition (3 disjoint blocks)
    try:
        U, V, W = _build_rank51()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def _build_rank4():
    """Build rank-4 decomposition for the delta(j,k) tensor.
    
    The tensor T[i,j,k] = delta(j,k) can be decomposed with one factor per j-value.
    For each j in {0,...,m-1}: U[i*m+j, j]=1, V[j*p+j, j]=1, W[j*n+i, j]=1 for all i.
    """
    n, m, p = DIMENSIONS
    rank = m  # 4
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for j in range(m):
        for i in range(n):
            U[i * m + j, j] = 1
        V[j * p + j, j] = 1
        for i in range(n):
            W[j * n + i, j] = 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank-4 delta tensor decomposition
    try:
        U, V, W = _build_rank4()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
