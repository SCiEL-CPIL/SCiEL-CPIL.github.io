import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Rank 49 Strassen-based decomposition
    # Based on the Kronecker product of two 2x2x2 Strassen decompositions
    rank = 49
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank)) # 16 x 49
    V = np.zeros((m * p, rank)) # 32 x 49
    W = np.zeros((p * n, rank)) # 32 x 49
    
    # Strassen 2x2x2 decomposition details
    # U_2, V_2, W_2 are 4x7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # A_specs: How to form A_eff for each P_k
    # A_eff is 2x2. Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    # B_specs: How to form B_eff for each P_k
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    # C_specs: How P_k contributes to C blocks
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    # Now, for each P_k (0..6) and each S_m (0..6):
    for k_p in range(7): # P_k
        if time.time() > deadline:
            break
        for m_s in range(7): # S_m
            col = k_p * 7 + m_s
            
            # --- U (Input A) ---
            for coeff_a, block_idx in A_specs[k_p]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_a_idx = global_r * 4 + global_c
                        a_eff_idx = r * 2 + c
                        U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
            
            # --- V (Input B) ---
            # B is 4x8. Split into Left (cols 0-3) and Right (cols 4-7)
            for coeff_b, block_idx in B_specs[k_p]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        b_eff_idx = r * 2 + c
                        val = coeff_b * V_2[b_eff_idx, m_s]
                        
                        # Left half (cols 0-3)
                        global_k_left = global_c
                        global_b_idx_left = global_r * 8 + global_k_left
                        V[global_b_idx_left, col] += val
                        
                        # Right half (cols 4-7)
                        global_k_right = global_c + 4
                        global_b_idx_right = global_r * 8 + global_k_right
                        V[global_b_idx_right, col] += val
            
            # --- W (Output C) ---
            # C is 4x8. W[k*n + i, r] => W[k*4 + i, r]
            # k is column index (0..7), i is row index (0..3)
            for coeff_c, c_block_idx in C_specs[k_p]:
                bi, bj = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        p_eff_idx = r * 2 + c
                        val = coeff_c * W_2[p_eff_idx, m_s]
                        
                        # Left half (cols 0-3)
                        global_k_left = bj * 2 + c
                        global_w_idx_left = global_k_left * 4 + global_i
                        W[global_w_idx_left, col] += val
                        
                        # Right half (cols 4-7)
                        global_k_right = bj * 2 + c + 4
                        global_w_idx_right = global_k_right * 4 + global_i
                        W[global_w_idx_right, col] += val
                        
            if time.time() > deadline:
                break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
