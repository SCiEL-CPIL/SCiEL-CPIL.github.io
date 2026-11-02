def construct():
    import time
    start_time = time.time()
    deadline = 100  # Safety margin
    
    n, m, p = 4, 4, 5
    # Start with naive rank 80 to ensure validity
    rank = n * m * p
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill naive decomposition
    # C[i,j] = sum_k A[i,k] B[k,j]
    # Term r corresponds to (i, k, j)
    # U index: i*m + k
    # V index: k*p + j
    # W index: j*n + i (based on prompt hint W[k*n+i] where k->j, i->i)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * m * p + k * p + j
                U[i*m + k][r] = 1
                V[k*p + j][r] = 1
                W[j*n + i][r] = 1
                
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
