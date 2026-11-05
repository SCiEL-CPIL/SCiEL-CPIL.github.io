"""Program to find rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns {'U': U, 'V': V, 'W': W} where U, V, W are lists of lists (or numpy arrays)
    representing the factor matrices of the tensor decomposition.
    
    The multiplication tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 if the multiplication tensor has a 1 at (i,j,k), else 0.
    
    For 3x3x7 matrix multiplication, the tensor is:
    T_{i,j,k} = 1 if i,j,k are valid indices (i in [0,3), j in [0,3), k in [0,7))
    Actually, the multiplication tensor for (n,m,p) matrices has entries:
    T_{i,j,k} = 1 for all i,j,k where the product A[i,j]*B[j,k] contributes to C[i,k].
    So the tensor is T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, that's the full tensor. Let me reconsider.
    
    The matrix multiplication tensor T_{n,m,p} has indices (i, j, k) where:
    - i indexes rows of A and C (0 to n-1)
    - j indexes columns of A and rows of B (0 to m-1)  
    - k indexes columns of B and C (0 to p-1)
    
    T_{i,j,k} = 1 for all valid i,j,k.
    
    A rank-r decomposition gives:
    sum_{c=1}^{r} U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)} = T_{i,j,k}
    
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Start with the schoolbook rank n*m*p = 63
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
                W[k*n + i, c] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
