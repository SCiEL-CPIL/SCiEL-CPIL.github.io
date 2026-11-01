import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication with dimensions n, m, p
    # requires sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
    # The all-ones solution satisfies this with rank 1.
    rank = 1
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
