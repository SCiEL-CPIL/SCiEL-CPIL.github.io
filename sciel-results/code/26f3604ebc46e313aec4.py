"""Matrix multiplication tensor decomposition for 2x4x7.

This program constructs an exact tensor decomposition of the matrix multiplication
tensor for multiplying a 2x4 matrix by a 4x7 matrix. The decomposition is based on
the schoolbook algorithm, which has rank n*m*p = 2*4*7 = 56.

The interface returns factor matrices U, V, W with integer entries such that
the matrix multiplication tensor is exactly decomposed.
"""

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the tensor decomposition for matrix multiplication 2x4 @ 4x7.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The decomposition satisfies: for all i, j, k,
    (A[i,:] dot B[:,k]) = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    where A is 2x4, B is 4x7, and the result C is 2x7 with C[i,k] = sum_j A[i,j]*B[j,k].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2 * 4 * 7 = 56
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (14, 56)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term corresponds to A[i,j] * B[j,k]
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
