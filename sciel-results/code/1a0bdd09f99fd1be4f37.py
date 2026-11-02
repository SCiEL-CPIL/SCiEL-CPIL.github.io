import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    The rank is 27 (schoolbook algorithm).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term corresponds to one multiplication A[i,j] * B[j,k] contributing to C[i,k]
                # U is indexed by i*m + j
                U[i * m + j][column] = 1
                # V is indexed by j*p + k
                V[j * p + k][column] = 1
                # W is indexed by k*n + i (Trace convention W[k*n+i])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
