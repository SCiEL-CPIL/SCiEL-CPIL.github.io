import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    
    This implementation uses the schoolbook algorithm, which achieves rank R = n*m*p = 60.
    The factors U, V, W are constructed such that the tensor 
    T[i, j, k, l] = delta_{j,k} delta_{i,l} (for (i,j) in A, (j,k) in B, (k,i) in C)
    is decomposed as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    
    The indexing convention for the output matrices is:
      U has shape (n*m, R). Row index = i*m + j.
      V has shape (m*p, R). Row index = j*p + k.
      W has shape (p*n, R). Row index = k*n + i.
      
    For the schoolbook decomposition, each column r corresponds to a unique triple (i, j, k).
    U[i*m+j, r] = 1 if r maps to (i, j, *)
    V[j*p+k, r] = 1 if r maps to (*, j, k)
    W[k*n+i, r] = 1 if r maps to (*, *, i)
    
    Specifically, for column r associated with triple (i, j, k):
      U[i*m + j, r] = 1
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1
    """
    n, m, p = DIMENSIONS
    R = n * m * p
    
    # Initialize U, V, W as numpy arrays of zeros with integer dtype
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
