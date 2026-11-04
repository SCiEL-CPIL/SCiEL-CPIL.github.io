import numpy as np
import time

def construct():
    """
    Construct an exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    
    Uses the standard schoolbook decomposition with rank n*m*p = 4*4*5 = 80.
    The decomposition is:
      C[i,k] = sum_j A[i,j] * B[j,k]
    which corresponds to tensor T[i,j,k] = 1 for all i,j,k.
    
    Factor matrices:
      U has shape (n*m, rank) = (16, 80)
      V has shape (m*p, rank) = (20, 80)  
      W has shape (p*n, rank) = (20, 80)
    
    For each (i,j,k), assign rank index r = i*p*m + j*p + k (or similar bijection).
    Then:
      U[i*m+j, r] = 1
      V[j*p+k, r] = 1
      W[k*n+i, r] = 1
    """
    n, m, p = 4, 4, 5
    rank = n * m * p  # 80
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, r] = 1.0
                # V index: j*p + k
                V[j * p + k, r] = 1.0
                # W index: k*n + i (following W[k*n+i, r] convention)
                W[k * n + i, r] = 1.0
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
