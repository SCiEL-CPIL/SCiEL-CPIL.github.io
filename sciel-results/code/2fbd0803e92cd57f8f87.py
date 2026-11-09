def construct():
    n, m, p = 2, 3, 7
    NM, MP, PN = n * m, m * p, p * n  # 6, 21, 14
    
    # Rank 1: all-ones decomposition
    # T_{i,j,k} = 1 = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0]
    # with U=V=W=1 (all entries 1)
    rank = 1
    U = [[1] * rank for _ in range(NM)]
    V = [[1] * rank for _ in range(MP)]
    W = [[1] * rank for _ in range(PN)]
    
    return {"U": U, "V": V, "W": W}
