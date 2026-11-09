import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    Currently uses the schoolbook algorithm with rank n*m*p = 30.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
                
    # Convert to lists of lists as required by the interface if strict, 
    # but the prompt says "factor matrices of shapes...", usually numpy arrays or lists are fine.
    # The seed program returned lists of lists. I will return lists of lists to be safe.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
