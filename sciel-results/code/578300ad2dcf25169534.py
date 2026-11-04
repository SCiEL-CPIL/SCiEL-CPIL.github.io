"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    import time
    _t0 = time.time()
    _deadline = _t0 + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Schoolbook decomposition: rank = n*m*p = 144
    # U[i*m+j, r] = 1 if (i,j,k) corresponds to rank-1 term r
    # V[j*p+k, r] = 1 if (i,j,k) corresponds to rank-1 term r  
    # W[k*n+i, r] = 1 if (i,j,k) corresponds to rank-1 term r
    # where C[k,i] = sum_j A[i,j] * B[j,k]
    
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
# EVOLVE-BLOCK-END
