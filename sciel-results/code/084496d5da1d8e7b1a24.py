"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 2x5x5 matrix multiplication.
    
    We decompose T_{2,5,5} as the direct sum of two 1x5x5 tensors.
    Each 1x5x5 tensor computes one row of the 5x5 product: C[k] = sum_j A[j]*B[j,k].
    The schoolbook rank for each 1x5x5 is 25, giving total rank 50.
    
    U: shape (n*m, rank) = (10, 50)
    V: shape (m*p, rank) = (25, 50)  
    W: shape (p*n, rank) = (10, 50)
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
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
