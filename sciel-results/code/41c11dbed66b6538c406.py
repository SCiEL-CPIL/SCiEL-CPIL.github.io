import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (24, rank)
    
    terms = []
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        
        A indices: rows i0, i0+1; cols j0, j0+1
        B indices: rows j0, j0+1; cols k0, k0+1
        C indices: rows i0, i0+1; cols k0, k0+1
        
        U index: (i, j) -> i*m + j
        V index: (j, k) -> j*p + k
        W index: (k, i) -> k*n + i
        """
        a11_u = i0 * m + j0
        a12_u = i0 * m + (j0 + 1)
        a21_u = (i0 + 1) * m + j0
        a22_u = (i0 + 1) * m + (j0 + 1)
        
        b11_v = j0 * p + k0
        b12_v = j0 * p + (k0 + 1)
        b21_v = (j0 + 1) * p + k0
        b22_v = (j0 + 1) * p + (k0 + 1)
        
        c11_w = k0 * n + i0
        c12_w = (k0 + 1) * n + i0
        c21_w = k0 * n + (i0 + 1)
        c22_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a11 + a22) * (b11 + b22) -> C11 += M1, C22 += M1
        u1 = np.zeros(n * m); u1[a11_u] = 1.0; u1[a22_u] = 1.0
        v1 = np.zeros(m * p); v1[b11_v] = 1.0; v1[b22_v] = 1.0
        w1 = np.zeros(p * n); w1[c11_w] = 1.0; w1[c22_w] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a21 + a22) * b11 -> C21 += M2, C22 -= M2
        u2 = np.zeros(n * m); u2[a21_u] = 1.0; u2[a22_u] = 1.0
        v2 = np.zeros(m * p); v2[b11_v] = 1.0
        w2 = np.zeros(p * n); w2[c21_w] = 1.0; w2[c22_w] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a11 * (b12 - b22) -> C12 += M3, C22 += M3
        u3 = np.zeros(n * m); u3[a11_u] = 1.0
        v3 = np.zeros(m * p); v3[b12_v] = 1.0; v3[b22_v] = -1.0
        w3 = np.zeros(p * n); w3[c12_w] = 1.0; w3[c22_w] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a22 * (b21 - b11) -> C11 += M4, C21 += M4
        u4 = np.zeros(n * m); u4[a22_u] = 1.0
        v4 = np.zeros(m * p); v4[b21_v] = 1.0; v4[b11_v] = -1.0
        w4 = np.zeros(p * n); w4[c11_w] = 1.0; w4[c21_w] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a11 + a12) * b22 -> C11 -= M5, C12 += M5
        u5 = np.zeros(n * m); u5[a11_u] = 1.0; u5[a12_u] = 1.0
        v5 = np.zeros(m * p); v5[b22_v] = 1.0
        w5 = np.zeros(p * n); w5[c11_w] = -1.0; w5[c12_w] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a21 - a11) * (b11 + b12) -> C22 += M6
        u6 = np.zeros(n * m); u6[a21_u] = 1.0; u6[a11_u] = -1.0
        v6 = np.zeros(m * p); v6[b11_v] = 1.0; v6[b12_v] = 1.0
        w6 = np.zeros(p * n); w6[c22_w] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a12 - a22) * (b21 + b22) -> C11 += M7
        u7 = np.zeros(n * m); u7[a12_u] = 1.0; u7[a22_u] = -1.0
        v7 = np.zeros(m * p); v7[b21_v] = 1.0; v7[b22_v] = 1.0
        w7 = np.zeros(p * n); w7[c11_w] = 1.0
        terms.append((u7, w7))
        # Fix: terms.append((u7, v7, w7))
        terms[-1] = (u7, v7, w7)
    
    def add_schoolbook_1x4_4x4(i_row, j0, k0):
        """
        Add schoolbook decomposition for A[i_row, j0:j0+4] * B[j0:j0+4, k0:k0+4]
        """
        for j in range(4):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = np.zeros(n * m); u_vec[u_idx] = 1.0
                v_vec = np.zeros(m * p); v_vec[v_idx] = 1.0
                w_vec = np.zeros(p * n); w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    
    # A1*B1: 4 blocks of 2x2*2x2
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(i0=0, j0=0, k0=0)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(i0=0, j0=2, k0=0)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(i0=0, j0=0, k0=2)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(i0=0, j0=2, k0=2)
    
    # A1*B2: 4 blocks of 2x2*2x2
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(i0=0, j0=0, k0=4)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(i0=0, j0=2, k0=4)
    # Block 7: A[0:2, 0:2] * B[0:2, 6:8]
    add_strassen_block(i0=0, j0=0, k0=6)
    # Block 8: A[0:2, 2:4] * B[2:4, 6:8]
    add_strassen_block(i0=0, j0=2, k0=6)
    
    # A2*B1: schoolbook 1x4*4x4
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=0)
    
    # A2*B2: schoolbook 1x4*4x4
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=4)
    
    rank = len(terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
