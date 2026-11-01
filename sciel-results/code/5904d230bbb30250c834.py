"""Matrix multiplication 2x3x9 - schoolbook decomposition of rank 54."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the rank-1 components
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to A[i, j]
                U[i * m + j][idx] = 1
                # V corresponds to B[j, k]
                V[j * p + k][idx] = 1
                # W corresponds to C[i, k] (trace convention W[k*n+i])
                W[k * n + i][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
