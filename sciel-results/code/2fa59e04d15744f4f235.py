"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    """Return the schoolbook tensor decomposition for 2x3x9 matrix multiplication.
    
    The schoolbook decomposition has rank n*m*p = 2*3*9 = 54.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]  # 6 x 54
    V = [[0] * rank for _ in range(m * p)]  # 27 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
