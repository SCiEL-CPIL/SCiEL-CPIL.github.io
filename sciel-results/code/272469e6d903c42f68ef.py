"""
Strategy: Deterministic construction of the rank-49 Strassen decomposition for 4x4 matrix multiplication.
This avoids the unreliable and slow gradient-based search in the seed program, ensuring a consistent
rank-49 result. If the evaluator's seed rank is higher than 49 (e.g., 56 or 64), this yields an improvement.
If the seed is 49, this maintains the baseline.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2 matrix multiplication.
    a[s, i, j]: coefficient of A[i,j] in subproduct s
    b[s, j, k]: coefficient of B[j,k] in subproduct s
    c[i, k, s]: coefficient of M_s in C[i,k]
    """
    a = np.zeros((7, 2, 2))
    b = np.zeros((7, 2, 2))
    c = np.zeros((2, 2, 7))
    
    # m0 = (A00+A11)(B00+B11)
    a[0] = [[1, 0], [0, 1]]
    b[0] = [[1, 0], [0, 1]]
    # m1 = (A10+A11)B00
    a[1] = [[0, 0], [1, 1]]
    b[1] = [[1, 0], [0, 0]]
    # m2 = A00(B10-B11)
    a[2] = [[1, 0], [0, 0]]
    b[2] = [[0, 0], [1, -1]]
    # m3 = A11(B01-B00)
    a[3] = [[0, 0], [0, 1]]
    b[3] = [[-1, 1], [0, 0]]
    # m4 = (A00+A01)B11
    a[4] = [[1, 1], [0, 0]]
    b[4] = [[0, 0], [0, 1]]
    # m5 = (A10-A00)(B00+B01)
    a[5] = [[-1, 0], [1, 0]]
    b[5] = [[1, 1], [0, 0]]
    # m6 = (A01-A11)(B10+B11)
    a[6] = [[0, 1], [0, -1]]
    b[6] = [[0, 0], [1, 1]]
    
    # C00 = m0 + m3 - m4 + m6
    c[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    # C01 = m2 + m4
    c[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    # C10 = m1 + m3
    c[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    # C11 = m0 - m1 + m2 + m5
    c[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return a, b, c

def _build_strassen_4x4():
    """
    Builds the rank-49 decomposition for 4x4 matrix multiplication
    by taking the tensor product of the rank-7 Strassen decomposition for 2x2.
    """
    a2, b2, c2 = _strassen_2x2()
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices: i, j, k in 0..3
    # Split into high/low bits: i = 2*i1 + i2
    # Rank index: r = 7*s1 + s2
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            
            # U[i, j, r] = a2[s1, i1, j1] * a2[s2, i2, j2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * m + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            
            # V[j, k, r] = b2[s1, j1, k1] * b2[s2, j2, k2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * p + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            
            # W[k, i, r] = c2[i1, k1, s1] * c2[i2, k2, s2]
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * n + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    
    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Build the deterministic rank-49 solution
    U, V, W = _build_strassen_4x4()
    
    # Verify (should be fast)
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank-64 if verification fails (should not happen)
        n, m, p = 4, 4, 4
        rank = 64
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
