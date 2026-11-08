def construct():
    n = n_dim
    m = m_dim
    p = p_dim
    
    # The tensor for matrix multiplication in this formulation is all ones.
    # T[i,j,k] = 1 for all i,j,k.
    # This tensor has rank 1: T = u (x) v (x) w where u,v,w are all-ones vectors.
    rank = 1
    
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
