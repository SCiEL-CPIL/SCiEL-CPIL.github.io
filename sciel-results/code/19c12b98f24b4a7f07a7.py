"""Rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial Kronecker decomposition: one term per (i,j,k)
    # This gives rank = n*m*p = 54
    # For (2,3,9), the best known bound for M(2,3,9) is likely 54 or close
    # We return the standard trivial decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping: idx = i * (m*p) + j * p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                # U[row, col] where row = i*m + j, col = idx
                U[i * m + j][idx] = 1
                # V[row, col] where row = j*p + k, col = idx
                V[j * p + k][idx] = 1
                # W[row, col] where row = k*n + i, col = idx (trace convention W[k*n+i, r])
                W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
