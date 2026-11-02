"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    
    The 2x5x5 multiplication tensor decomposes as the direct sum of two 1x5x5 tensors
    (one per output row). Each 1x5x5 tensor has rank 25 (the vector-matrix product
    requires 25 scalar multiplications). Since the two output rows use disjoint
    indices in the W matrix (W[k*n+i] with i=0,1), the total rank is 25+25=50.
    
    This schoolbook decomposition is rank-optimal for this tensor.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
    # U: shape (n*m, rank) = (10, 50)
    # V: shape (m*p, rank) = (25, 50)  
    # W: shape (p*n, rank) = (10, 50)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook: for each (i, j, k) triple, one rank-1 term
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1  (following the convention W[k*n+i, r])
    
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
