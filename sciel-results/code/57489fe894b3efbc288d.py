"""Optimized search for a rank-minimal exact tensor decomposition of 3x3 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# Known rank for 3x3 matrix multiplication is 23 (Landsberg).
# We will attempt to find a decomposition of rank 23 or lower.



# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses Strassen's 7-term algorithm for the 2x2 core and optimized border terms.
    Attempts rank 19, then falls back to higher ranks.
    """
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_r = U.reshape((n, m, R))
        V_r = V.reshape((m, p, R))
        W_r = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    def build_strassen(U, V, W):
        """
        Build the 7 Strassen terms in columns 0-6.
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B11-B10)
        M4 = A11(B00-B10)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 - M2 + M3 - M7
        C01 = M5 + M7
        C10 = M2 + M4
        C11 = M1 - M3 - M4 + M6
        """
        # M1 = (A00+A11)(B00+B11)
        r = 0
        U[idx_u(0, 0), r] = 1
        U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        V[idx_v(1, 1), r] = 1
        # C00 += M1, C11 += M1
        W[idx_w(0, 0), r] = 1
        W[idx_w(1, 1), r] = 1
        
        # M2 = (A10+A11)B00
        r = 1
        U[idx_u(1, 0), r] = 1
        U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        # C10 += M2, C00 -= M2
        W[idx_w(0, 1), r] = 1
        W[idx_w(0, 0), r] = -1
        
        # M3 = A00(B11-B10)
        r = 2
        U[idx_u(0, 0), r] = 1
        V[idx_v(1, 1), r] = 1
        V[idx_v(1, 0), r] = -1
        # C00 += M3, C11 -= M3
        W[idx_w(0, 0), r] = 1
        W[idx_w(1, 1), r] = -1
        
        # M4 = A11(B00-B10)
        r = 3
        U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        V[idx_v(1, 0), r] = -1
        # C10 += M4, C11 -= M4
        W[idx_w(0, 1), r] = 1
        W[idx_w(1, 1), r] = -1
        
        # M5 = (A00+A01)B11
        r = 4
        U[idx_u(0, 0), r] = 1
        U[idx_u(0, 1), r] = 1
        V[idx_v(1, 1), r] = 1
        # C01 += M5
        W[idx_w(1, 0), r] = 1
        
        # M6 = (A10-A00)(B00+B01)
        r = 5
        U[idx_u(1, 0), r] = 1
        U[idx_u(0, 0), r] = -1
        V[idx_v(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1
        # C11 += M6
        W[idx_w(1, 1), r] = 1
        
        # M7 = (A01-A11)(B10+B11)
        r = 6
        U[idx_u(0, 1), r] = 1
        U[idx_u(1, 1), r] = -1
        V[idx_v(1, 0), r] = 1
        V[idx_v(1, 1), r] = 1
        # C01 += M7, C00 -= M7
        W[idx_w(1, 0), r] = 1
        W[idx_w(0, 0), r] = -1
    
    # --- Try rank 19: Strassen(7) + 12 optimized border terms ---
    R = 19
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    build_strassen(U, V, W)
    
    col = 7
    # i=2 border terms (7 terms)
    # a20*(b00+b01) -> c20, c21
    U[idx_u(2, 0), col] = 1
    V[idx_v(0, 0), col] = 1
    V[idx_v(0, 1), col] = 1
    W[idx_w(0, 2), col] = 1
    W[idx_w(1, 2), col] = 1
    col += 1
    # a20*b02 -> c22
    U[idx_u(2, 0), col] = 1
    V[idx_v(0, 2), col] = 1
    W[idx_w(2, 2), col] = 1
    col += 1
    # a21*(b10+b11) -> c20, c21
    U[idx_u(2, 1), col] = 1
    V[idx_v(1, 0), col] = 1
    V[idx_v(1, 1), col] = 1
    W[idx_w(0, 2), col] = 1
    W[idx_w(1, 2), col] = 1
    col += 1
    # a21*b12 -> c22
    U[idx_u(2, 1), col] = 1
    V[idx_v(1, 2), col] = 1
    W[idx_w(2, 2), col] = 1
    col += 1
    # a22*b20 -> c20
    U[idx_u(2, 2), col] = 1
    V[idx_v(2, 0), col] = 1
    W[idx_w(0, 2), col] = 1
    col += 1
    # a22*b21 -> c21
    U[idx_u(2, 2), col] = 1
    V[idx_v(2, 1), col] = 1
    W[idx_w(1, 2), col] = 1
    col += 1
    # a22*b22 -> c22
    U[idx_u(2, 2), col] = 1
    V[idx_v(2, 2), col] = 1
    W[idx_w(2, 2), col] = 1
    col += 1
    # j=2 border terms (3 terms)
    # (a00+a10)*b20 -> c02, c12
    U[idx_u(0, 0), col] = 1
    U[idx_u(1, 0), col] = 1
    V[idx_v(2, 0), col] = 1
    W[idx_w(2, 0), col] = 1
    W[idx_w(2, 1), col] = 1
    col += 1
    # (a01+a11)*b21 -> c02, c12
    U[idx_u(0, 1), col] = 1
    U[idx_u(1, 1), col] = 1
    V[idx_v(2, 1), col] = 1
    W[idx_w(2, 0), col] = 1
    W[idx_w(2, 1), col] = 1
    col += 1
    # (a02+a12)*b22 -> c02, c12
    U[idx_u(0, 2), col] = 1
    U[idx_u(1, 2), col] = 1
    V[idx_v(2, 2), col] = 1
    W[idx_w(2, 0), col] = 1
    W[idx_w(2, 1), col] = 1
    col += 1
    # k=2 border terms (2 terms)
    # (a00+a10)*b02 -> c02, c12
    U[idx_u(0, 0), col] = 1
    U[idx_u(1, 0), col] = 1
    V[idx_v(0, 2), col] = 1
    W[idx_w(2, 0), col] = 1
    W[idx_w(2, 1), col] = 1
    col += 1
    # (a01+a11)*b12 -> c02, c12
    U[idx_u(0, 1), col] = 1
    U[idx_u(1, 1), col] = 1
    V[idx_v(1, 2), col] = 1
    W[idx_w(2, 0), col] = 1
    W[idx_w(2, 1), col] = 1
    col += 1
    
    if verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # --- Try rank 23: Strassen(7) + 16 border terms ---
    R = 23
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    build_strassen(U, V, W)
    
    col = 7
    # 16 border terms with some combining
    # i=2: a20*(b00+b01), a20*b02, a21*(b10+b11), a21*b12, a22*b20, a22*b21, a22*b22 (7)
    U[idx_u(2, 0), col] = 1; V[idx_v(0, 0), col] = 1; V[idx_v(0, 1), col] = 1; W[idx_w(0, 2), col] = 1; W[idx_w(1, 2), col] = 1; col += 1
    U[idx_u(2, 0), col] = 1; V[idx_v(0, 2), col] = 1; W[idx_w(2, 2), col] = 1; col += 1
    U[idx_u(2, 1), col] = 1; V[idx_v(1, 0), col] = 1; V[idx_v(1, 1), col] = 1; W[idx_w(0, 2), col] = 1; W[idx_w(1, 2), col] = 1; col += 1
    U[idx_u(2, 1), col] = 1; V[idx_v(1, 2), col] = 1; W[idx_w(2, 2), col] = 1; col += 1
    U[idx_u(2, 2), col] = 1; V[idx_v(2, 0), col] = 1; W[idx_w(0, 2), col] = 1; col += 1
    U[idx_u(2, 2), col] = 1; V[idx_v(2, 1), col] = 1; W[idx_w(1, 2), col] = 1; col += 1
    U[idx_u(2, 2), col] = 1; V[idx_v(2, 2), col] = 1; W[idx_w(2, 2), col] = 1; col += 1
    # j=2: (a00+a10)*b20, (a01+a11)*b21, (a02+a12)*b22 (3)
    U[idx_u(0, 0), col] = 1; U[idx_u(1, 0), col] = 1; V[idx_v(2, 0), col] = 1; W[idx_w(0, 0), col] = 1; W[idx_w(0, 1), col] = 1; col += 1
    U[idx_u(0, 1), col] = 1; U[idx_u(1, 1), col] = 1; V[idx_v(2, 1), col] = 1; W[idx_w(1, 0), col] = 1; W[idx_w(1, 1), col] = 1; col += 1
    U[idx_u(0, 2), col] = 1; U[idx_u(1, 2), col] = 1; V[idx_v(2, 2), col] = 1; W[idx_w(2, 0), col] = 1; W[idx_w(2, 1), col] = 1; col += 1
    # k=2: a00*b02, a01*b12, a10*b02, a11*b12 (4) - no combining possible
    U[idx_u(0, 0), col] = 1; V[idx_v(0, 2), col] = 1; W[idx_w(2, 0), col] = 1; col += 1
    U[idx_u(0, 1), col] = 1; V[idx_v(1, 2), col] = 1; W[idx_w(2, 0), col] = 1; col += 1
    U[idx_u(1, 0), col] = 1; V[idx_v(0, 2), col] = 1; W[idx_w(2, 1), col] = 1; col += 1
    U[idx_u(1, 1), col] = 1; V[idx_v(1, 2), col] = 1; W[idx_w(2, 1), col] = 1; col += 1
    # That's 7+3+4=14 border terms, total 21. Need 2 more for rank 23.
    # Actually let me just use 16 separate border terms for rank 23.
    
    if verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # --- Fallback: rank 26 (Strassen 7 + 19 separate border terms) ---
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    build_strassen(U, V, W)
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
    
    if verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # --- Final fallback: schoolbook rank 27 ---
    R = 27
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
