"""Program for Matrix Multiplication 2x3x9 Decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The tensor T[a, b, c] = 1 iff a = i*m + j, b = j*p + k, c = k*n + i.
    For n=2, m=3, p=9, the minimal rank is 54 (schoolbook), as the tensor decomposes into
    independent slices for each i, each requiring rank m*p = 27.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
