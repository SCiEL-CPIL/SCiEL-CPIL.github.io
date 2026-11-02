import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen decomposition for 2x2 matrix multiplication
    # We define the coefficients for U, V, W for the 7 rank-1 terms.
    # U is 4x7, V is 4x7, W is 4x7 (local indices)
    
    # Local indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    u1 = {0: 1, 3: 1}
    v1 = {0: 1, 3: 1}
    w1 = {0: 1, 3: 1}
    
    # Term 2: M2 = (A10 + A11)B00
    u2 = {2: 1, 3: 1}
    v2 = {0: 1}
    w2 = {2: 1, 3: -1}
    
    # Term 3: M3 = A00(B01 - B11)
    u3 = {0: 1}
    v3 = {1: 1, 3: -1}
    w3 = {1: 1, 3: 1}
    
    # Term 4: M4 = A11(B10 - B00)
    u4 = {3: 1}
    v4 = {2: 1, 0: -1}
    w4 = {0: 1, 2: 1}
    
    # Term 5: M5 = (A00 + A01)B11
    u5 = {0: 1, 1: 1}
    v5 = {3: 1}
    w5 = {0: -1, 1: 1}
    
    # Term 6: M6 = (A10 - A00)(B00 + B01)
    u6 = {2: 1, 0: -1}
    v6 = {0: 1, 1: 1}
    w6 = {3: 1}
    
    # Term 7: M7 = (A01 - A11)(B10 + B11)
    u7 = {1: 1, 3: -1}
    v7 = {2: 1, 3: 1}
    w7 = {0: 1}
    
    terms_u = [u1, u2, u3, u4, u5, u6, u7]
    terms_v = [v1, v2, v3, v4, v5, v6, v7]
    terms_w = [w1, w2, w3, w4, w5, w6, w7]
    
    # Map local indices to global indices for the 3 blocks
    # Block b corresponds to columns 2b and 2b+1 of B, and rows 2b and 2b+1 of C (transposed in W index logic? No, W index is k*n+i)
    # C is 2x6. Output C[i, k].
    # W index: k*n + i.
    
    for b in range(3):
        for t in range(7):
            r_idx = b * 7 + t
            u_dict = terms_u[t]
            v_dict = terms_v[t]
            w_dict = terms_w[t]
            
            # Fill U
            for a_idx, val in u_dict.items():
                U[a_idx, r_idx] = val
            
            # Fill V
            # Local B idx -> Global V idx
            # 0: (0, 2b) -> 2b
            # 1: (0, 2b+1) -> 2b+1
            # 2: (1, 2b) -> 6 + 2b
            # 3: (1, 2b+1) -> 7 + 2b
            for l_idx, val in v_dict.items():
                if l_idx == 0:
                    g_idx = 2 * b
                elif l_idx == 1:
                    g_idx = 2 * b + 1
                elif l_idx == 2:
                    g_idx = 6 + 2 * b
                elif l_idx == 3:
                    g_idx = 7 + 2 * b
                V[g_idx, r_idx] = val
            
            # Fill W
            # Local C idx -> Global W idx
            # Local C indices are (i, k_local) where k_local is 0 or 1.
            # But my w_dict keys are flattened local C indices:
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            # Wait, my w_dict keys above were:
            # w1: {0: 1, 3: 1} -> C(0,0) and C(1,1)
            # w2: {2: 1, 3: -1} -> C(1,0) and C(1,1)
            # w3: {1: 1, 3: 1} -> C(0,1) and C(1,1)
            # w4: {0: 1, 2: 1} -> C(0,0) and C(1,0)
            # w5: {0: -1, 1: 1} -> C(0,0) and C(0,1)
            # w6: {3: 1} -> C(1,1)
            # w7: {0: 1} -> C(0,0)
            
            # Let's verify these keys correspond to (i, k_local) flattened as i*2 + k_local?
            # 0: (0,0) -> 0
            # 1: (0,1) -> 1
            # 2: (1,0) -> 2
            # 3: (1,1) -> 3
            # Yes.
            
            for c_l_idx, val in w_dict.items():
                # c_l_idx is flattened (i, k_local)
                i = c_l_idx // 2
                k_local = c_l_idx % 2
                k_global = 2 * b + k_local
                w_g_idx = k_global * n + i
                W[w_g_idx, r_idx] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
