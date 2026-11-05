"""
Rewritten program for 4x4x4 matrix multiplication tensor decomposition.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2x2
to achieve a rank of 49.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (4, 7).
    
    Indices:
    U[i*2+j, r] = coeff of A[i,j]
    V[j*2+k, r] = coeff of B[j,k]
    W[k*2+i, r] = coeff of C[i,k] (Trace convention W[k*n+i, r])
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _trivial_decomp(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly computes matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    u_val = U[i * m + j, r]
                    v_val = V[j * p + k, r]
                    w_val = W[k * n + i, r]
                    if u_val != 0.0 and v_val != 0.0 and w_val != 0.0:
                        s += u_val * v_val * w_val
                expected = 1.0
                if abs(s - expected) > tol:
                    return False, (i, j, k, s)
    return True, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Get base Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Construct 4x4 decomposition via Kronecker product
    # Rank = 7 * 7 = 49
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indices for 4x4:
    # i = 2*i1 + i2
    # j = 2*j1 + j2
    # k = 2*k1 + k2
    # r = 7*r1 + r2
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    u_idx = i * 4 + j
                    for k1 in range(2):
                        for k2 in range(2):
                            k = 2 * k1 + k2
                            v_idx = j * 4 + k
                            w_idx = k * 4 + i
                            
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    u_val = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                                    v_val = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                                    w_val = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
                                    
                                    U[u_idx, r] = u_val
                                    V[v_idx, r] = v_val
                                    W[w_idx, r] = w_val
                                    
                                    if time.time() > deadline:
                                        return _trivial_decomp(n, m, p)
    
    # Verify correctness
    if time.time() < deadline - 5.0:
        ok, info = _verify(U, V, W, n, m, p)
        if not ok:
            print(f"Verification failed at {info}")
            return _trivial_decomp(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
