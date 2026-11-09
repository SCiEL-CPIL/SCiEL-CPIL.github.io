"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the matrix multiplication tensor decomposition.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if i,j,k correspond to the product, else 0.
    Specifically, the tensor T_{i,j,k} = 1 if the indices align with the multiplication, which is always 1 in the standard basis.
    Wait, the tensor for matrix multiplication A(3x5) * B(5x5) = C(3x5) is defined such that
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor T_{i,j,k} = 1 for all valid i, j, k.
    The decomposition is T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c ? 
    No, the standard form is:
    T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c is not quite right for the indices.
    
    Standard Tensor Decomposition for MatMul:
    T_{i,j,k} = 1 (for all i,j,k)
    Decomposition: T_{i,j,k} = sum_c u_{i,j}^c v_{j,k}^c w_{k,i}^c
    This is incorrect. The indices for U, V, W in the prompt are:
    U: (n*m, rank) -> indexed by (i, j)
    V: (m*p, rank) -> indexed by (j, k)
    W: (p*n, rank) -> indexed by (k, i)
    
    The condition is:
    sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1 for all i, j, k.
    This is the standard schoolbook decomposition where each term picks out one (i,j,k) triplet.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use lists of lists as per the seed program to ensure compatibility, 
    # though numpy arrays are often accepted if converted. 
    # The seed used lists. I will stick to lists to be safe, or convert numpy to list.
    
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
