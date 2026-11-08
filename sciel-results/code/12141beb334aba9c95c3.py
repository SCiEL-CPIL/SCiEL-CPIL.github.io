import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal (schoolbook) tensor decomposition for 2x3 * 3x7 matrix multiplication.
    The rank is n*m*p = 42.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A indices (i, j) to column col
                U[i * m + j, col] = 1.0
                # V maps input B indices (j, k) to column col
                V[j * p + k, col] = 1.0
                # W maps output C indices (k, i) to column col
                # Note: The task specifies W[k*n+i, r]. 
                # Standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
                # The tensor index for output is usually (i, k).
                # The prompt says "output factor follows the official trace convention W[k*n+i,r]".
                # This implies the index in W is k*n + i.
                # Let's verify the index mapping.
                # If C is 2x7, indices are (i, k) with i in [0,1], k in [0,6].
                # Linear index for C is usually i*p + k (row-major) or k*n + i (col-major).
                # The prompt explicitly says W[k*n+i, r].
                # So for output element C[i, k], the row in W is k*n + i.
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
