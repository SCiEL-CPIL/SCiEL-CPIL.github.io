def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # The minimal rank for multiplying (n x m) by (m x p) is m.
    # We can achieve this by summing over the m intermediate dimensions.
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        # c corresponds to the j-th intermediate index
        for i in range(n):
            # U[i*m + c, c] = 1
            U[i * m + c][c] = 1
            
            for k in range(p):
                # V[c*p + k, c] = 1
                V[c * p + k][c] = 1
                # W[k*n + i, c] = 1
                W[k * n + i][c] = 1
                
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_decomposition(result, n, m, p):
        return result
    else:
        # Fallback
        return _schoolbook_decomposition(n, m, p)
