def construct():
    n, m, p = 4, 4, 5
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        i0 = c // p
        k0 = c % p
        
        # U[i*m + j, c] = 1 if i == i0
        for j in range(m):
            U[i0 * m + j][c] = 1
            
        # V[j*p + k, c] = 1 if k == k0
        for j in range(m):
            V[j * p + k0][c] = 1
            
        # W[k*n + i, c] = 1 if k == k0 and i == i0
        W[k0 * n + i0][c] = 1
        
    return {"U": U, "V": V, "W": W}
