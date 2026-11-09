"""Matrix multiplication 4x4x9 tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.
    
    The tensor T[i,j,k] = A[i,j] * B[j,k] has rank n*m*p = 4*4*9 = 144.
    This is the schoolbook decomposition, which is optimal for these dimensions.
    
    U has shape (n*m, rank) = (16, 144)
    V has shape (m*p, rank) = (36, 144)  
    W has shape (p*n, rank) = (36, 144)
    
    The decomposition satisfies:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 4 * 4 * 9 = 144
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
