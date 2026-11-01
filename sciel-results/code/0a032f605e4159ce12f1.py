"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition of rank n*m*p for 2x3*3x8 matrix multiplication.
    
    The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    for all i in range(n), j in range(m), k in range(p).
    
    We decompose it as sum of rank-1 tensors:
      T = sum_c U[:,c] * V[:,c] * W[:,c]
    
    For schoolbook: each (i,j,k) triple gives one rank-1 term.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2*3*8 = 48
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
