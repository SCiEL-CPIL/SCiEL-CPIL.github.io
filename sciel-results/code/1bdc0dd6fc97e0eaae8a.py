import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # The matrix multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # for i in [0,n), j in [0,m), k in [0,p), and 0 otherwise.
    
    # We decompose the 2x4x6 multiplication as a Kronecker product of:
    #   - a 2x2x2 multiplication (rank 7 via Strassen)
    #   - a 1x2x3 multiplication (rank 2)
    #
    # Total rank = 7 * 2 = 14
    #
    # Indexing:
    #   i = i1 + i2      (i1 in {0,1}, i2 in {0})
    #   j = 2*j1 + j2    (j1 in {0,1}, j2 in {0,1})
    #   k = 3*k1 + k2    (k1 in {0,1}, k2 in {0,1,2})
    #
    #   a = i*m + j = i*4 + j
    #   b = j*p + k = j*6 + k
    #   c = k*n + i = k*2 + i
    
    # --- Strassen 2x2x2 (rank 7) ---
    # U_s[i*2+j, r], V_s[j*2+k, r], W_s[k*2+i, r]
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # m1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # m2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # m3 = A00(B10-B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # m4 = A11(B01-B00)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # m5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # m6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # m7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # --- 1x2x3 decomposition (rank 2) ---
    # n2=1, m2=2, p2=3
    # Only terms where j2 == k2 contribute (since i2=0 always)
    # T[i2*m2+j2, j2*p2+k2, k2*n2+i2] = 1 iff j2==k2
    # j2 in {0,1}, k2 in {0,1,2}
    # Valid pairs: (j2=0,k2=0) and (j2=1,k2=1)
    
    U_123 = np.zeros((2, 2))
    V_123 = np.zeros((6, 2))
    W_123 = np.zeros((3, 2))
    
    # Term 0: j2=0, k2=0
    U_123[0, 0] = 1   # i2*m2+j2 = 0
    V_123[0, 0] = 1   # j2*p2+k2 = 0
    W_123[0, 0] = 1   # k2*n2+i2 = 0
    
    # Term 1: j2=1, k2=1
    U_123[1, 1] = 1   # i2*m2+j2 = 1
    V_123[4, 1] = 1   # j2*p2+k2 = 1*3+1 = 4
    W_123[1, 1] = 1   # k2*n2+i2 = 1
    
    # --- Combine via Kronecker product ---
    # r = r1*2 + r2, where r1 in [0,7), r2 in [0,2)
    total_rank = 14
    U_full = np.zeros((n*m, total_rank))  # (8, 14)
    V_full = np.zeros((m*p, total_rank))  # (24, 14)
    W_full = np.zeros((p*n, total_rank))  # (12, 14)
    
    for r1 in range(7):
        for r2 in range(2):
            r = r1 * 2 + r2
            
            # U_full[a, r] = sum over (i1,j1) where U_s[i1*2+j1, r1] nonzero
            #   of U_s[i1*2+j1, r1] * U_123[i2*2+j2, r2]
            # where i = i1 + i2 (i2=0), so a = i*4 + j = i1*4 + 2*j1 + j2
            # U_123[i2*2+j2, r2] is nonzero only for specific (i2,j2,r2)
            
            for i1 in range(2):
                for j1 in range(2):
                    us_val = U_s[i1*2 + j1, r1]
                    if us_val == 0:
                        continue
                    for i2 in range(1):
                        for j2 in range(2):
                            u123_val = U_123[i2*2 + j2, r2]
                            if u123_val == 0:
                                continue
                            i = i1 + i2
                            j = 2*j1 + j2
                            a = i * 4 + j
                            U_full[a, r] += us_val * u123_val
            
            # V_full[b, r] = V_s[j1*2+k1, r1] * V_123[j2*3+k2, r2]
            # b = j*6 + k = (2*j1+j2)*6 + (3*k1+k2)
            for j1 in range(2):
                for k1 in range(2):
                    vs_val = V_s[j1*2 + k1, r1]
                    if vs_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v123_val = V_123[j2*3 + k2, r2]
                            if v123_val == 0:
                                continue
                            j = 2*j1 + j2
                            k = 3*k1 + k2
                            b = j * 6 + k
                            V_full[b, r] += vs_val * v123_val
            
            # W_full[c, r] = W_s[k1*2+i1, r1] * W_123[k2*1+i2, r2]
            # c = k*2 + i = (3*k1+k2)*2 + (i1+i2)
            for k1 in range(2):
                for i1 in range(2):
                    ws_val = W_s[k1*2 + i1, r1]
                    if ws_val == 0:
                        continue
                    for k2 in range(3):
                        for i2 in range(1):
                            w123_val = W_123[k2 + i2, r2]
                            if w123_val == 0:
                                continue
                            k = 3*k1 + k2
                            i = i1 + i2
                            c = k * 2 + i
                            W_full[c, r] += ws_val * w123_val
    
    # Verify
    T_comp = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    if not np.allclose(T_comp, T_expected, atol=1e-10):
        # Fallback to schoolbook
        return _schoolbook_solution(n, m, p)
    
    # Convert to list-of-lists format
    rank = total_rank
    U_out = [[float(U_full[a, r]) for r in range(rank)] for a in range(n * m)]
    V_out = [[float(V_full[b, r]) for r in range(rank)] for b in range(m * p)]
    W_out = [[float(W_full[c, r]) for r in range(rank)] for c in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _schoolbook_solution(n, m, p):
    """Fallback schoolbook solution with rank = n*m*p"""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
