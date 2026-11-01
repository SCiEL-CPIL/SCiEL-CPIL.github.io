"""Matrix multiplication 2x2 * 2x13 rank-minimal exact tensor decomposition."""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p  # 26
    
    U = [[0] * rank for _ in range(n * m)]  # (n*m, rank) = (4, 26)
    V = [[0] * rank for _ in range(m * p)]  # (m*p, rank) = (26, 26)
    W = [[0] * rank for _ in range(p * n)]  # (p*n, rank) = (26, 26)
    
    for k in range(p):
        # Two rank-1 terms for this k: r = 2k and r = 2k + 1
        for j in range(m):
            # V: row index is j*p + k
            V[j * p + k][2 * k] = 1 if j == 0 else 0
            V[j * p + k][2 * k + 1] = 1 if j == 1 else 0
            
            # U: row index is i*m + j, for all i
            for i in range(n):
                U[i * m + j][2 * k] = 1
                U[i * m + j][2 * k + 1] = 1
        
        # W: row index is k*n + i, for all i
        for i in range(n):
            W[k * n + i][2 * k] = 1
            W[k * n + i][2 * k + 1] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
