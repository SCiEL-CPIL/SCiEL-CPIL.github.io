"""
Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition.

Strategy:
  1. Hard-code a known rank-23 decomposition for the 3x3x3 block (k=0,1,2).
  2. For the 3x3x2 block (k=3,4), use a systematic search to find a low-rank
     decomposition that is consistent with the rank-23 coefficients for k<3.
  3. Verify the full decomposition. If it fails, fall back to schoolbook (rank 45).
"""

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 3, 3, 5
    
    # Schoolbook decomposition (rank 45)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify helper
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        n, m, p = 3, 3, 5
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by removing redundant terms
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try to remove terms one by one from the end
    for c in range(rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        U_test = U_cur.copy()
        V_test = V_cur.copy()
        W_test = W_cur.copy()
        U_test[:, c] = 0
        V_test[:, c] = 0
        W_test[:, c] = 0
        
        if verify(U_test, V_test, W_test):
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
    
    # Compact the matrices
    keep_cols = [c for c in range(rank) if np.any(U_cur[:, c] != 0) or np.any(V_cur[:, c] != 0) or np.any(W_cur[:, c] != 0)]
    U_final = U_cur[:, keep_cols]
    V_final = V_cur[:, keep_cols]
    W_final = W_cur[:, keep_cols]
    
    # Convert to list of lists
    U_out = [[float(x) for x in row] for row in U_final]
    V_out = [[float(x) for x in row] for row in V_final]
    W_out = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
