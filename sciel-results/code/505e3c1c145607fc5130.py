import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    _start_time = time.time()
    _deadline = _start_time + 120
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    
    terms = []
    
    def time_check():
        return time.time() > _deadline
        
    def add_strassen_block(i0, j0, k0):
        if time_check():
            return
            
        # Indices for A (n x m)
        a00_u = i0 * m + j0
        a01_u = i0 * m + (j0 + 1)
        a10_u = (i0 + 1) * m + j0
        a11_u = (i0 + 1) * m + (j0 + 1)
        
        # Indices for B (m x p)
        b00_v = j0 * p + k0
        b01_v = j0 * p + (k0 + 1)
        b10_v = (j0 + 1) * p + k0
        b11_v = (j0 + 1) * p + (k0 + 1)
        
        # Indices for C (p x n) -> W[k*n + i]
        c00_w = k0 * n + i0
        c01_w = (k0 + 1) * n + i0
        c10_w = k0 * n + (i0 + 1)
        c11_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = [0.0] * nm; u1[a00_u] = 1.0; u1[a11_u] = 1.0
        v1 = [0.0] * mp; v1[b00_v] = 1.0; v1[b11_v] = 1.0
        w1 = [0.0] * pn; w1[c00_w] = 1.0; w1[c11_w] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a10 + a11) * b00
        u2 = [0.0] * nm; u2[a10_u] = 1.0; u2[a11_u] = 1.0
        v2 = [0.0] * mp; v2[b00_v] = 1.0
        w2 = [0.0] * pn; w2[c10_w] = 1.0; w2[c11_w] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a00 * (b01 - b11)
        u3 = [0.0] * nm; u3[a00_u] = 1.0
        v3 = [0.0] * mp; v3[b01_v] = 1.0; v3[b11_v] = -1.0
        w3 = [0.0] * pn; w3[c01_w] = 1.0; w3[c11_w] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a11 * (b10 - b00)
        u4 = [0.0] * nm; u4[a11_u] = 1.0
        v4 = [0.0] * mp; v4[b10_v] = 1.0; v4[b00_v] = -1.0
        w4 = [0.0] * pn; w4[c00_w] = 1.0; w4[c10_w] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a00 + a01) * b11
        u5 = [0.0] * nm; u5[a00_u] = 1.0; u5[a01_u] = 1.0
        v5 = [0.0] * mp; v5[b11_v] = 1.0
        w5 = [0.0] * pn; w5[c00_w] = -1.0; w5[c01_w] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a10 - a00) * (b00 + b01)
        u6 = [0.0] * nm; u6[a10_u] = 1.0; u6[a00_u] = -1.0
        v6 = [0.0] * mp; v6[b00_v] = 1.0; v6[b01_v] = 1.0
        w6 = [0.0] * pn; w6[c11_w] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a01 - a11) * (b10 + b11)
        u7 = [0.0] * nm; u7[a01_u] = 1.0; u7[a11_u] = -1.0
        v7 = [0.0] * mp; v7[b10_v] = 1.0; v7[b11_v] = 1.0
        w7 = [0.0] * pn; w7[c00_w] = 1.0
        terms.append((u7, v7, w7))

    def add_schoolbook_1x4_4x4(i_row, j0, k0):
        if time_check():
            return
        for j in range(4):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = [0.0] * nm; u_vec[u_idx] = 1.0
                v_vec = [0.0] * mp; v_vec[v_idx] = 1.0
                w_vec = [0.0] * pn; w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))

    # A1 (2x4) * B1 (4x4)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    
    # A1 (2x4) * B2 (4x4)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 0, 6)
    add_strassen_block(0, 2, 4)
    add_strassen_block(0, 2, 6)
    
    # A2 (1x4) * B1 (4x4)
    add_schoolbook_1x4_4x4(2, 0, 0)
    
    # A2 (1x4) * B2 (4x4)
    add_schoolbook_1x4_4x4(2, 0, 4)

    rank = len(terms)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    
    # Verify decomposition correctness
    if not time_check():
        # Build the tensor from decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Compare with true tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+c, c=i*p+j... 
        # Actually for 3x4*4x8: A[i,j]*B[j,k] -> C[i,k], W[k*n+i]
        # T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
        # T is nonzero (=1) when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        
        T_true = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_true[a_idx, b_idx, c_idx] = 1.0
        
        err = np.max(np.abs(T_decomp - T_true))
        if err > 1e-9:
            # Fallback to schoolbook if verification fails
            terms_fb = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        u_vec = [0.0] * nm; u_vec[a_idx] = 1.0
                        v_vec = [0.0] * mp; v_vec[b_idx] = 1.0
                        w_vec = [0.0] * pn; w_vec[c_idx] = 1.0
                        terms_fb.append((u_vec, v_vec, w_vec))
            rank_fb = len(terms_fb)
            U_fb = np.zeros((nm, rank_fb))
            V_fb = np.zeros((mp, rank_fb))
            W_fb = np.zeros((pn, rank_fb))
            for r in range(rank_fb):
                u_vec, v_vec, w_vec = terms_fb[r]
                U_fb[:, r] = u_vec
                V_fb[:, r] = v_vec
                W_fb[:, r] = w_vec
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
