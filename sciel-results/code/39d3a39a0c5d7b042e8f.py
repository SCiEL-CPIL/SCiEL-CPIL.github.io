import numpy as np

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # P2 = (A10 + A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # P3 = A00(B01 - B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # P4 = A11(B10 - B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # P5 = (A00 + A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition.
    Rank 49.
    """
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen terms for 4x4
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Effective A and B for each of the 7 top-level P_k
    A_eff_specs = [
        [(1, 0), (1, 3)], # P1: A00+A11
        [(1, 2), (1, 3)], # P2: A10+A11
        [(1, 0)],         # P3: A00
        [(1, 3)],         # P4: A11
        [(1, 0), (1, 1)], # P5: A00+A01
        [(1, 2), (-1, 0)],# P6: A10-A00
        [(1, 1), (-1, 3)] # P7: A01-A11
    ]
    
    B_eff_specs = [
        [(1, 0), (1, 3)], # P1: B00+B11
        [(1, 0)],         # P2: B00
        [(1, 1), (-1, 3)],# P3: B01-B11
        [(1, 2), (-1, 0)],# P4: B10-B00
        [(1, 3)],         # P5: B11
        [(1, 0), (1, 1)], # P6: B00+B01
        [(1, 2), (1, 3)]  # P7: B10+B11
    ]
    
    C_eff_specs = [
        [(1, 0), (1, 3)], # C00+C11
        [(1, 2), (-1, 3)],# C10-C11
        [(1, 1), (1, 3)], # C01+C11
        [(1, 0), (1, 2)], # C00+C10
        [(-1, 0), (1, 1)],# -C00+C01
        [(1, 3)],         # C11
        [(1, 0)]          # C00
    ]
    
    for k in range(7):
        for m in range(7):
            col_idx = k * 7 + m
            
            # U contribution
            for coeff_a, a_blk in A_eff_specs[k]:
                bi, bj = divmod(a_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        g_idx = gr * 4 + gc
                        l_idx = lr * 2 + lc
                        U[g_idx, col_idx] += coeff_a * U_2[l_idx, m]
                        
            # V contribution
            for coeff_b, b_blk in B_eff_specs[k]:
                bi, bj = divmod(b_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        g_idx = gr * 4 + gc
                        l_idx = lr * 2 + lc
                        V[g_idx, col_idx] += coeff_b * V_2[l_idx, m]
                        
            # W contribution
            for coeff_c, c_blk in C_eff_specs[k]:
                bi, bj = divmod(c_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        # W index for 4x4 is col*4 + row
                        g_idx = gc * 4 + gr
                        l_idx = lr * 2 + lc
                        W[g_idx, col_idx] += coeff_c * W_2[l_idx, m]
                        
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Get 4x4x4 decomposition
    U_4, V_4, W_4 = _strassen_4x4x4()
    
    # Apply to Left half (B cols 0-3)
    for i in range(16):
        U[i, 0:49] = U_4[i, :]
        
    for i in range(16):
        r, c = divmod(i, 4)
        v_idx_left = r * 8 + c
        V[v_idx_left, 0:49] = V_4[i, :]
        
    for i in range(16):
        W[i, 0:49] = W_4[i, :]
        
    # Apply to Right half (B cols 4-7)
    for i in range(16):
        U[i, 49:98] = U_4[i, :]
        
    for i in range(16):
        r, c = divmod(i, 4)
        v_idx_right = r * 8 + (c + 4)
        V[v_idx_right, 49:98] = V_4[i, :]
        
    for i in range(16):
        w_idx_global = i + 16
        W[w_idx_global, 49:98] = W_4[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
