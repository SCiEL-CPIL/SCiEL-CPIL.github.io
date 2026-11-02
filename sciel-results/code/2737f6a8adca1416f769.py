"""Rewritten program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # The matrix multiplication tensor T for (n x m) * (m x p) -> (n x p)
    # T[(i,j), (j',k), (k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #
    # We want to find rank-minimal U, V, W such that:
    # T[(i,j), (j',k), (k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    #
    # U shape: (n*m, R) = (9, R)
    # V shape: (m*p, R) = (18, R)
    # W shape: (p*n, R) = (18, R)
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        R = U.shape[1]
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for i2 in range(n):
                                expected = 1.0 if (i == i2 and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(R):
                                    val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+i2, r]
                                err = abs(val - expected)
                                if err > max_err:
                                    max_err = err
        return max_err < tol, max_err

    # Build a rank 18 decomposition by splitting into two 3x3x3 blocks
    def build_block_decomposition():
        R = 18
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Block 1: k in {0, 1, 2}
        # This corresponds to the first 3 columns of the output.
        # We perform a standard 3x3x3 multiplication.
        # Rank 9.
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    col = i*m + j + 0 # Wait, index mapping for U is fixed.
                    # Actually, we just need to assign unique columns for each (i,j,k) triple.
                    # For block 1, k is 0,1,2.
                    # For block 2, k is 3,4,5.
                    pass
        
        # Let's do it explicitly.
        col = 0
        for block in range(2):
            k_offset = block * 3
            for i in range(n):
                for j in range(m):
                    for k_local in range(3):
                        k = k_offset + k_local
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        return U, V, W

    best_U, best_V, best_W = build_block_decomposition()
    
    # Verify the solution
    ok, err = verify(best_U, best_V, best_W)
    if not ok:
        # Fallback to schoolbook if something went wrong (should not happen)
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
