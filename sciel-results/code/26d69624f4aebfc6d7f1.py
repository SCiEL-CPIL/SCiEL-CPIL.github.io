"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-96 exact decomposition for 3x4 * 4x8 matrix multiplication.
    
    The tensor for (A,B) -> C where A is 3x4, B is 4x8, C is 3x8:
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    We use the schoolbook decomposition with rank = n*m*p = 3*4*8 = 96.
    U has shape (n*m, rank) = (12, 96)
    V has shape (m*p, rank) = (32, 96)
    W has shape (p*n, rank) = (24, 96)
    
    For each (i,j,k) triple, we create one column c where:
      U[i*m + j, c] = 1
      V[j*p + k, c] = 1
      W[k*n + i, c] = 1
    
    This gives: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 96
    
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
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 100:  # Well within 120s limit
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
