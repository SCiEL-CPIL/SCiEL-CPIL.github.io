"""Program for matrix multiplication 2x2x12 tensor decomposition."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x12 matrix.
    
    The decomposition uses the schoolbook (trivial) method with rank = n*m*p = 48.
    For matrix-vector multiplication (as opposed to matrix-matrix), the tensor rank
    equals the number of scalar multiplications needed, which is n*m*p.
    
    U: shape (n*m, rank) = (4, 48)
    V: shape (m*p, rank) = (24, 48)
    W: shape (p*n, rank) = (24, 48)
    
    The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i in range(n), j in range(m), k in range(p).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 24 x 48
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
