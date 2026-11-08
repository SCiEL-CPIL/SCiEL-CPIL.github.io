import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    rank = 98

    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

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

    # Block Strassen coefficients for 4x4x4
    # P_k = sum_{blocks} coeff * A_block * B_block
    # C_block = sum_{P_k} coeff * P_k

    # P_k definitions (A_blocks, B_blocks)
    # P0: (A00+A11)(B00+B11)
    # P1: (A10+A11)B00
    # P2: A00(B01-B11)
    # P3: A11(B10-B00)
    # P4: (A00+A01)B11
    # P5: (A10-A00)(B00+B01)
    # P6: (A01-A11)(B10+B11)

    # A_specs and B_specs as in original
    A_specs = [
        [(1, 0), (1, 3)], # P0: A00 + A11
        [(1, 2), (1, 3)], # P1: A10 + A11
        [(1, 0)],         # P2: A00
        [(1, 3)],         # P3: A11
        [(1, 0), (1, 1)], # P4: A00 + A01
        [(1, 2), (-1, 0)],# P5: A10 - A00
        [(1, 1), (-1, 3)],# P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0: B00 + B11
        [(1, 0)],         # P1: B00
        [(1, 1), (-1, 3)],# P2: B01 - B11
        [(1, 2), (-1, 0)],# P3: B10 - B00
        [(1, 3)],         # P4: B11
        [(1, 0), (1, 1)], # P5: B00 + B01
        [(1, 2), (1, 3)], # P6: B10 + B11
    ]
    
    C_specs = [
        # P0
        [(1, 0), (1, 3)], # C00 + C11
        # P1
        [(1, 2), (-1, 3)], # C10 - C11
        # P2
        [(1, 1), (1, 3)], # C01 + C11
        # P3
        [(1, 0), (1, 2)], # C00 + C10
        # P4
        [(-1, 0), (1, 1)], # -C00 + C01
        # P5
        [(1, 3)], # C11
        # P6
        [(1, 0)], # C00
    ]

    # Iterate to build U, V, W
    for half in range(2): # 0 for L, 1 for R
        for k_p in range(7): # P_k
            for m_s in range(7): # S_m
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            val = coeff_a * U_2[a_eff_idx, m_s]
                            if val != 0:
                                U[global_a_idx, col] += val

                # --- V (Input B) ---
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            val = coeff_b * V_2[b_eff_idx, m_s]
                            if val != 0:
                                V[global_b_idx, col] += val

                # --- W (Output C) ---
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            val = coeff_c * W_2[p_eff_idx, m_s]
                            if val != 0:
                                W[global_w_idx, col] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
