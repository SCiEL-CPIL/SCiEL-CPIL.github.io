"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-64 exact schoolbook decomposition for 2x4 * 4x8 matrix multiplication.
    
    The matrix multiplication tensor for C[i,k] = sum_j A[i,j]*B[j,k]
    has a natural rank-nmp decomposition where each term corresponds
    to a single (i,j,k) triple.
    
    U has shape (n*m, rank) = (8, 64)
    V has shape (m*p, rank) = (32, 64)  
    W has shape (p*n, rank) = (16, 64)
    
    W uses the official trace convention W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2 * 4 * 8 = 64
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
