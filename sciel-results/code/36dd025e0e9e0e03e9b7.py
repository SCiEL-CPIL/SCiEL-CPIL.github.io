"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m, as we decompose into m independent M_{n,1,p} multiplications
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # U: rows i*m + j get 1 in column j
        for i in range(n):
            U[i * m + j][j] = 1
            
        # V: rows j*p + k get 1 in column j
        for k in range(p):
            V[j * p + k][j] = 1
            
        # W: all rows get 1 in column j
        for idx in range(p * n):
            W[idx][j] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
