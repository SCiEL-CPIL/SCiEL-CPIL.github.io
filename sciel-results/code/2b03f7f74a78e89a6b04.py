"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    We use the schoolbook decomposition of rank n*m*p = 54.
    For each (i, j, k) with i in [0,n), j in [0,m), k in [0,p):
      U[i*m + j, r] = 1
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1
    where r is the column index corresponding to (i, j, k).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
