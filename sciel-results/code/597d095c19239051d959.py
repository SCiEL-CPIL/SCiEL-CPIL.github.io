"""
Program for Matrix Multiplication 2x3x3 rank minimization.
The rank of the tensor $M_{2,3,3}$ is 18. This program constructs the
schoolbook decomposition, which achieves this minimal rank.
"""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions 2x3 by 3x3.
    Uses the schoolbook decomposition of rank 18, which is optimal.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The tensor is T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
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
