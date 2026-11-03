def _build_4x4x5():
    n, m, p = DIMENSIONS  # 4, 4, 5
    base = _build_4x4x4_strassen()
    rank_base = 49
    rank = rank_base + 4
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 20 x rank
    W = [[0] * rank for _ in range(p * n)]  # 20 x rank
    
    # Copy 4x4x4 part with proper index remapping
    # Base U: index i*4+j (i,j in 0..3) -> same in new problem
    for i in range(4):
        for j in range(4):
            for r in range(rank_base):
                U[i*4 + j][r] = base["U"][i*4 + j][r]
    
    # Base V: index j*4+k (j,k in 0..3) -> new index j*5+k
    for j in range(4):
        for k in range(4):
            for r in range(rank_base):
                V[j*5 + k][r] = base["V"][j*4 + k][r]
    
    # Base W: index k*4+i (k,i in 0..3) -> same in new problem (n=4 unchanged)
    for k in range(4):
        for i in range(4):
            for r in range(rank_base):
                W[k*4 + i][r] = base["W"][k*4 + i][r]
    
    # Add 4x4x1 part for column k=4
    for i in range(4):
        for j in range(4):
            r = rank_base + i
            U[i*4 + j][r] = 1
            V[j*5 + 4][r] = 1
            W[4*4 + i][r] = 1  # k=4, so k*n+i = 16+i
    
    return {"U": U, "V": V, "W": W}
