"""Matrix multiplication 3x4x6 with tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 72
    # Each term is A[i,j] * B[j,k] contributing to C[i,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i (per trace convention W[k*n+i, r])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
