"""
Program to construct a rank-minimal exact tensor decomposition for matrix multiplication.
Target: 3x4 matrix by 4x4 matrix.
Dimensions: n=3, m=4, p=4.
Schoolbook rank: 3*4*4 = 48.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for matrix multiplication C = A * B.
    
    Shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta (reconstructs C_ik)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1.0
                # V index: j*p + k
                V[j * p + k][col] = 1.0
                # W index: k*n + i
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
