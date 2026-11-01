"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m * p
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We decompose by (j, k) pairs.
    # For each k (output column) and j (input column of A), we have a term.
    # The term contributes to C[i, k] for all i.
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    col = 0
    for k in range(p):
        for j in range(m):
            # Set V
            V[j * p + k][col] = 1
            
            # Set U and W for all i
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
