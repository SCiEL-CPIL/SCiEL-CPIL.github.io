import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 if the entry (i,k) of A@B is computed
    # from A[i,j] * B[j,k]. For standard matrix mult, T[i,j,k] = 1 for all valid indices.
    # We want to find rank-1 decompositions: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i,j], v_r[j,k], w_r[k,i]
    
    # Start with the full tensor
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T = np.ones((n, m, p), dtype=object)
    
    # We'll try to peel off rank-1 components
    # A rank-1 component is defined by vectors u (shape n*m), v (shape m*p), w (shape p*n)
    # such that T_remaining[i,j,k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # Strategy: Try to find rank-1 components by looking at the structure
    # For matrix multiplication, the tensor is very structured.
    
    # Let's try a systematic approach: iterate over possible rank-1 patterns
    # and try to find if they are valid (i.e., all entries are integers or half-integers)
    
    # First, let's just use schoolbook as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m + j) * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
    
    # Now try to find better decompositions using peeling
    # The idea: if we can find a rank-1 component that exactly matches part of the tensor,
    # we can remove it and reduce the rank needed.
    
    # For now, return schoolbook as it's guaranteed to work
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
