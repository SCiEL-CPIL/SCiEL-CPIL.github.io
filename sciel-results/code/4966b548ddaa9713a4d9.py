"""Matrix multiplication tensor decomposition for 2x4 * 4x6."""

def construct():
    n, m, p = 2, 4, 6
    rank = m * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 24)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 24)
    
    # Index r corresponds to (j, k) pair: r = j * p + k
    # V[j*p+k, r] = 1 if r == j*p+k (identity)
    for j in range(m):
        for k in range(p):
            r = j * p + k
            V[j * p + k][r] = 1
    
    # U[i*m+j, r] and W[k*n+i, r] for r = j*p+k
    # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Since V is identity, only r = j*p+k contributes:
    # U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # Set U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1 for all i,j,k
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
