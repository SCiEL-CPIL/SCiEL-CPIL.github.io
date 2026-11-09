<<<<<<< SEARCH
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' (list-of-lists) and rank = number of columns.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_UVW = (U_sb, V_sb, W_sb)
    
    # Try block decomposition
    U_bd, V_bd, W_bd = _block_decompose(n, m, p)
    if U_bd.shape[1] < best_rank:
        best_rank = U_bd.shape[1]
        best_UVW = (U_bd, V_bd, W_bd)
    
    # Verify the best decomposition
    if not _verify(best_UVW[0], best_UVW[1], best_UVW[2], n, m, p):
        # Fallback to schoolbook if verification fails
        best_UVW = (U_sb, V_sb, W_sb)
        best_rank = n * m * p
    
    # Try optimization
    if time.time() - _START_TIME < _DEADLINE - 10:
        U_opt, V_opt, W_opt = _try_optimize(best_UVW[0], best_UVW[1], best_UVW[2], n, m, p)
        if U_opt.shape[1] < best_rank and _verify(U_opt, V_opt, W_opt, n, m, p):
            best_rank = U_opt.shape[1]
            best_UVW = (U_opt, V_opt, W_opt)
    
    return {
        "U": best_UVW[0].tolist(),
        "V": best_UVW[1].tolist(),
        "W": best_UVW[2].tolist()
    }
=======
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' (list-of-lists) and rank = number of columns.
    """
    n, m, p = DIMENSIONS
    
    # The tensor for dense matrix multiplication is the all-ones tensor.
    # T[i, j, k] = 1 for all i, j, k.
    # This tensor has rank 1.
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Sum = 1*1*1 = 1.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    # Verify to be safe
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something is wrong with the assumption
        U, V, W = _schoolbook(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
>>>>>>> REPLACE
