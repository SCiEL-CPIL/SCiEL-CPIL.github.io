"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication of a 2x3 matrix
    by a 3x4 matrix.
    
    The tensor M_{n,m,p} is decomposed into rank r = n*m*p.
    U has shape (n*m, r)
    V has shape (m*p, r)
    W has shape (p*n, r)
    
    The index mapping is:
    U[i*m + j, c] corresponds to input A[i, j]
    V[j*p + k, c] corresponds to input B[j, k]
    W[k*n + i, c] corresponds to output C[i, k]
    
    Each rank-1 term c multiplies A[i,j] * B[j,k] and adds to C[i,k].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m + j
                # V index: row j*p + k
                # W index: row k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
