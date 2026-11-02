"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # Strategy: Use Strassen decomposition for the two 2x2x2 blocks.
    # This gives rank 14. We then attempt to reduce rank by checking
    # if we can merge terms across blocks using shared structure.
    # However, the standard Strassen decomposition for 2x2x2 has rank 7.
    # Two independent blocks give rank 14.
    # Can we do better?
    # The tensor for 2x2x4 multiplication has rank at most 14.
    # Lower bounds suggest rank >= 14 for 2x2x4 (since 2x2x2 has rank 7, and the problem decomposes).
    # Actually, the rank of the 2x2x4 tensor is known to be 14.
    # So the minimal rank is 14.
    
    # Therefore, the current Strassen-based rank-14 decomposition is optimal.
    # We just need to construct it correctly and efficiently.
    
    rank = 14
    U = np.zeros((u_dim, rank), dtype=float)
    V = np.zeros((v_dim, rank), dtype=float)
    W = np.zeros((w_dim, rank), dtype=float)
    
    # Indices:
    # A[i,j] -> U[i*m + j]
    # B[j,k] -> V[j*p + k]
    # C[k,i] -> W[k*n + i]  (trace convention: W[k*n+i, r])
    
    # A indices:
    # a00=0, a01=1, a10=2, a11=3
    u00, u01, u10, u11 = 0, 1, 2, 3
    
    # B indices for block 1 (k=0,1):
    # b00=0, b01=1, b10=4, b11=5
    # B indices for block 2 (k=2,3):
    # b02=2, b03=3, b12=6, b13=7
    v00, v01, v02, v03 = 0, 1, 2, 3
    v10, v11, v12, v13 = 4, 5, 6, 7
    
    # C indices (W):
    # C[0,0]=0, C[1,0]=1  (k=0)
    # C[0,1]=2, C[1,1]=3  (k=1)
    # C[0,2]=4, C[1,2]=5  (k=2)
    # C[0,3]=6, C[1,3]=7  (k=3)
    w00, w10 = 0, 1
    w01, w11 = 2, 3
    w02, w12 = 4, 5
    w03, w13 = 6, 7
    
    # Strassen terms for 2x2x2 block with B columns (b1, b2) and C columns (c1, c2)
    # M1 = (a00+a11)(b1+b2) -> contributes to c1 (w0) and c2 (w1)
    # M2 = (a10+a11)b1 -> contributes to c1 (w10) and c2 (w11 with -1)
    # M3 = a00(b2-b1) -> wait, standard Strassen:
    # Let's use the standard 7-term Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3
    
    def add_strassen_block(b_col0, b_col1, c_col0, c_col1, r_start):
        """
        b_col0, b_col1: indices in V for B[0, col0], B[0, col1], B[1, col0], B[1, col1]
        Actually, V indices are:
        B[0, col0] = v0 + col0
        B[0, col1] = v0 + col1
        B[1, col0] = v0 + p + col0
        B[1, col1] = v0 + p + col1
        
        c_col0, c_col1: indices in W for C[0, col0], C[1, col0], C[0, col1], C[1, col1]
        W indices are:
        C[0, col0] = c_col0 * n + 0
        C[1, col0] = c_col0 * n + 1
        C[0, col1] = c_col1 * n + 0
        C[1, col1] = c_col1 * n + 1
        """
        r = r_start
        
        # B indices
        vb00 = b_col0       # B[0, col0]
        vb01 = b_col1       # B[0, col1]
        vb10 = p + b_col0   # B[1, col0]
        vb11 = p + b_col1   # B[1, col1]
        
        # W indices
        wc00 = c_col0 * n + 0  # C[0, col0]
        wc10 = c_col0 * n + 1  # C[1, col0]
        wc01 = c_col1 * n + 0  # C[0, col1]
        wc11 = c_col1 * n + 1  # C[1, col1]
        
        # M1 = (a00+a11)(b00+b11)
        U[u00, r] += 1
        U[u11, r] += 1
        V[vb00, r] += 1
        V[vb11, r] += 1
        W[wc00, r] += 1
        W[wc11, r] += 1
        r += 1
        
        # M2 = (a10+a11)b00
        U[u10, r] += 1
        U[u11, r] += 1
        V[vb00, r] += 1
        W[wc10, r] += 1
        W[wc11, r] -= 1
        r += 1
        
        # M3 = a00(b01-b11)
        U[u00, r] += 1
        V[vb01, r] += 1
        V[vb11, r] -= 1
        W[wc01, r] += 1
        W[wc11, r] += 1
        r += 1
        
        # M4 = a11(b10-b00)
        U[u11, r] += 1
        V[vb10, r] += 1
        V[vb00, r] -= 1
        W[wc00, r] += 1
        W[wc10, r] += 1
        r += 1
        
        # M5 = (a00+a01)b11
        U[u00, r] += 1
        U[u01, r] += 1
        V[vb11, r] += 1
        W[wc00, r] -= 1
        W[wc01, r] += 1
        r += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10, r] += 1
        U[u00, r] -= 1
        V[vb00, r] += 1
        V[vb01, r] += 1
        W[wc11, r] += 1
        r += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01, r] += 1
        U[u11, r] -= 1
        V[vb10, r] += 1
        V[vb11, r] += 1
        W[wc00, r] += 1
        r += 1
        
        return r
    
    # Block 1: columns 0, 1
    r = add_strassen_block(0, 1, 0, 1, 0)
    
    # Block 2: columns 2, 3
    r = add_strassen_block(2, 3, 2, 3, r)
    
    # Verify decomposition
    # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Should equal 1 if i=(a,b), j=(b,c), k=(c,a) for some a,b,c
    # i = a*m + b, j = b*p + c, k = c*n + a
    
    T_expected = np.zeros((u_dim, v_dim, w_dim), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_expected[i, j, k] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    if not np.allclose(T_recon, T_expected, atol=1e-10):
        # Fallback to schoolbook rank 16
        rank_fb = n * m * p
        U_fb = np.zeros((u_dim, rank_fb), dtype=float)
        V_fb = np.zeros((v_dim, rank_fb), dtype=float)
        W_fb = np.zeros((w_dim, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
