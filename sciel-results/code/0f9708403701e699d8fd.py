"""Optimized matrix multiplication decomposition for 2x5 * 5x6."""

def construct():
    n, m, p = 2, 5, 6
    
    # Schoolbook decomposition: rank = n*m*p = 60
    # U shape: (n*m, rank) = (10, 60)
    # V shape: (m*p, rank) = (30, 60)  
    # W shape: (p*n, rank) = (12, 60)
    # 
    # Tensor: T[i,j,k] = 1 for the (i,j,k) triple
    # U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t] summed over t gives C[i,k]
    #
    # For schoolbook: each (i,j,k) gets its own rank-1 term
    # U[i*m+j, i*m*p + j*p + k] = 1
    # V[j*p+k, i*m*p + j*p + k] = 1
    # W[k*n+i, i*m*p + j*p + k] = 1
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
