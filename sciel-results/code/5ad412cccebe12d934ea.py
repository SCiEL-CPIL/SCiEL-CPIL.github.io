"""
Matrix multiplication 6x6x6 rank-minimal exact tensor decomposition.
Current best known rank for 6x6 is 216 (schoolbook).
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank of the schoolbook algorithm is n*m*p
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j, c] = 1
                U[i * m + j][col] = 1
                # V[j, k, c] = 1
                V[j * p + k][col] = 1
                # W[k, i, c] = 1 (Note: Output factor follows W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
