# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    
    # Decompose into 6 independent rank-1 terms, one per (i,j) pair
    # For each (i,j), assign a unique rank index r = i*m + j
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    rank = n * m  # 6
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1.0
            for k in range(p):
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                assert abs(s - 1.0) < 1e-9, f"Verification failed at ({i},{j},{k}): {s}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
