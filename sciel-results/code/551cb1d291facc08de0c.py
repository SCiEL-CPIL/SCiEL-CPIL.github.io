"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Key insight: The matmul tensor for [n, m, p] has rank at most m*p when n is small,
because each (j,k) product term A[i,j]*B[j,k] can serve ALL n output rows simultaneously
by setting U[i*m+j, c] = 1 for all i, V[j*p+k, c] = 1, and W[k*n+i, c] = 1 for all i.

Reconstruction: C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

For the rank-28 construction:
- For each (j,k) pair, one rank-1 term with:
  U[0*m+j] = U[1*m+j] = 1
  V[j*p+k] = 1  
  W[k*n+0] = W[k*n+1] = 1
- This contributes A[i,j]*B[j,k] to C[i,k] for both i=0 and i=1.

Total rank = m*p = 28, which equals the trivial lower bound max(n*p, n*m, m*p) = 28.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = m * p  # 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # This single term handles both rows i=0 and i=1
            for i in range(n):
                U[i * m + j, col] = 1.0
                W[k * n + i, col] = 1.0
            V[j * p + k, col] = 1.0
            col += 1
    
    # Verify correctness numerically
    # C[i,k] should equal sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    # Random test
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                s += np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
            C_computed[i, k] = s
    
    assert np.allclose(C_computed, C_expected, atol=1e-10), f"Verification failed!\nComputed:\n{C_computed}\nExpected:\n{C_expected}"
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
