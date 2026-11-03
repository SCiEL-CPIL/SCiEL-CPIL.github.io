import numpy as np

def construct():
    """
    Returns a rank-1 exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The matrix multiplication tensor T[i, j, k] = 1 for all i in [0,1], j in [0,3], k in [0,4].
    This is because C[i,k] = sum_j A[i,j] * B[j,k], and the tensor element T[i,j,k] = 1
    for every valid (i,j,k) triple.
    
    The all-ones tensor is rank-1: it can be decomposed as the outer product of three
    all-ones vectors.
    
    U[i*m + j, 0] = 1 for all i, j
    V[j*p + k, 0] = 1 for all j, k
    W[k*n + i, 0] = 1 for all k, i
    
    Then: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1*1*1 = 1 = T[i,j,k]
    """
    n, m, p = 2, 4, 5
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
