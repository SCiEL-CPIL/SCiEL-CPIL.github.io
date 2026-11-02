"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    # Outer Strassen coefficients
    outer_A = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 0, 1, -1],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    
    outer_B = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    outer_C = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, C11(-1)
        [0, 1, 0, 1],   # M3 -> C01, C11
        [-1, 0, 1, 0],  # M4 -> C00(-1), C10
        [1, 1, 0, 0],   # M5 -> C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0]    # M7 -> C00
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x9_direct():
    """
    Direct construction of M_{4,4,9} decomposition.
    Split k into k0 (0..3) and k1 (4..8).
    k0 uses 4x4x4 Strassen (rank 49).
    k1 uses 4x4x5 with 4x4x4 Strassen (rank 49) + 4x4x1 schoolbook (rank 16).
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    total_rank = 114
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: k=0..3, 4x4x4 Strassen
    U1, V1, W1 = _decomp_4x4x4()
    U[:, 0:49] = U1
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V1[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W1[k*4 + i, :]
    
    # Part 2: k=4..7, 4x4x4 Strassen (for k1=0..3 of the 4x4x5)
    U2, V2, W2 = _decomp_4x4x4()
    U[:, 49:98] = U2
    for j in range(4):
        for k in range(4):
            V[j*9 + (4+k), 49:98] = V2[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(4+k)*4 + i, 49:98] = W2[k*4 + i, :]
    
    # Part 3: k=8, 4x4x1 schoolbook
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    U[:, 98:114] = U3
    for j in range(4):
        V[j*9 + 8, 98:114] = V3[j, :]
    for i in range(4):
        W[8*4 + i, 98:114] = W3[i, :]
    
    return U, V, W

def _check_decomp(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition using tensor contraction."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(r):
                    s += U3[i, j, t] * V3[j, k, t] * W3[k, i, t]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    limit = t0 + 130.0
    
    # Primary: direct construction with rank 114
    U, V, W = _decomp_4x4x9_direct()
    
    if time.time() < limit - 5.0:
        ok = _check_decomp(U, V, W, n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 144)
    t1 = time.time()
    if t1 < limit - 5.0:
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    # Last resort: minimal valid output
    r = n * m * p
    Uf = np.zeros((n*m, r))
    Vf = np.zeros((m*p, r))
    Wf = np.zeros((p*n, r))
    for c in range(r):
        i = c // (m*p)
        j = (c // p) % m
        k = c % p
        Uf[i*m + j, c] = 1
        Vf[j*p + k, c] = 1
        Wf[k*n + i, c] = 1
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

# EVOLVE-BLOCK-END
