import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_strassen_2x2x2(i0, j0, k0):
        # Adds 7 terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # Indices:
        # A: (i, j) -> i*m + j
        # B: (j, k) -> j*p + k
        # C: (k, i) -> k*n + i
        
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = [0.0] * (n * m); u1[a00] = 1.0; u1[a11] = 1.0
        v1 = [0.0] * (m * p); v1[b00] = 1.0; v1[b11] = 1.0
        w1 = [0.0] * (p * n); w1[c00] = 1.0; w1[c11] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a10 + a11) * b00
        u2 = [0.0] * (n * m); u2[a10] = 1.0; u2[a11] = 1.0
        v2 = [0.0] * (m * p); v2[b00] = 1.0
        w2 = [0.0] * (p * n); w2[c10] = 1.0; w2[c11] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a00 * (b01 - b11)
        u3 = [0.0] * (n * m); u3[a00] = 1.0
        v3 = [0.0] * (m * p); v3[b01] = 1.0; v3[b11] = -1.0
        w3 = [0.0] * (p * n); w3[c01] = 1.0; w3[c11] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a11 * (b10 - b00)
        u4 = [0.0] * (n * m); u4[a11] = 1.0
        v4 = [0.0] * (m * p); v4[b10] = 1.0; v4[b00] = -1.0
        w4 = [0.0] * (p * n); w4[c00] = 1.0; w4[c10] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a00 + a01) * b11
        u5 = [0.0] * (n * m); u5[a00] = 1.0; u5[a01] = 1.0
        v5 = [0.0] * (m * p); v5[b11] = 1.0
        w5 = [0.0] * (p * n); w5[c00] = -1.0; w5[c01] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a10 - a00) * (b00 + b01)
        u6 = [0.0] * (n * m); u6[a10] = 1.0; u6[a00] = -1.0
        v6 = [0.0] * (m * p); v6[b00] = 1.0; v6[b01] = 1.0
        w6 = [0.0] * (p * n); w6[c11] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a01 - a11) * (b10 + b11)
        u7 = [0.0] * (n * m); u7[a01] = 1.0; u7[a11] = -1.0
        v7 = [0.0] * (m * p); v7[b10] = 1.0; v7[b11] = 1.0
        w7 = [0.0] * (p * n); w7[c00] = 1.0
        terms.append((u7, v7, w7))

    def add_schoolbook_1x2x2(i_row, j0, k0):
        # Adds 4 terms for A[i_row, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        for j in range(2):
            for k in range(2):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = [0.0] * (n * m); u_vec[u_idx] = 1.0
                v_vec = [0.0] * (m * p); v_vec[v_idx] = 1.0
                w_vec = [0.0] * (p * n); w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))

    # We decompose 3x4x8 into 8 blocks of 3x2x2
    # j in {0, 2} (split 4 into 2+2)
    # k in {0, 2, 4, 6} (split 8 into 2+2+2+2)
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Top 2 rows of A (indices 0, 1)
            add_strassen_2x2x2(i0=0, j0=j0, k0=k0)
            # Bottom 1 row of A (index 2)
            add_schoolbook_1x2x2(i_row=2, j0=j0, k0=k0)
            
    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][r] = u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][r] = v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][r] = w_vec[k]
                
    return {"U": U, "V": V, "W": W}
