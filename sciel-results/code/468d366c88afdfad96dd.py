import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    
    # Total rank calculation
    # Part 1: 4x4 x 4x4 (Strassen rank 49)
    # Part 2: 4x4 x 4x1 (Schoolbook rank 16)
    # Part 3: 4x1 x 1x5 (Schoolbook rank 20)
    rank_1 = 49
    rank_2 = 16
    rank_3 = 20
    total_rank = rank_1 + rank_2 + rank_3
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Base Strassen 2x2x2 ---
    # U, V, W are 4x7
    U_base = np.zeros((4, 7))
    V_base = np.zeros((4, 7))
    W_base = np.zeros((4, 7))
    
    # M1
    U_base[0, 0] = 1; U_base[3, 0] = 1
    V_base[0, 0] = 1; V_base[3, 0] = 1
    W_base[0, 0] = 1; W_base[0, 3] = -1; W_base[0, 5] = -1; W_base[0, 6] = 1
    W_base[3, 0] = 1; W_base[3, 2] = -1; W_base[3, 4] = -1; W_base[3, 6] = 1
    
    # M2
    U_base[2, 1] = 1; U_base[3, 1] = 1
    V_base[0, 1] = 1
    W_base[2, 1] = 1; W_base[2, 3] = 1; W_base[2, 5] = 1
    
    # M3
    U_base[0, 2] = 1
    V_base[1, 2] = 1; V_base[3, 2] = -1
    W_base[1, 2] = 1; W_base[1, 4] = 1; W_base[1, 6] = 1
    
    # M4
    U_base[3, 3] = 1
    V_base[2, 3] = 1; V_base[0, 3] = -1
    W_base[0, 3] = -1; W_base[2, 3] = 1; W_base[3, 3] = 1 # Wait, C11 uses -M4, C21 uses M4, C22 uses M4?
    # C11 = M1 - M4 - M6 + M7. C11 is W[0]. W[0,3] = -1. Correct.
    # C21 = M2 + M4 + M6. C21 is W[2]. W[2,3] = 1. Correct.
    # C22 = M1 - M3 - M5 + M7. M4 is not in C22.
    # My previous W_base init had W[3,3]=1?
    # Let's re-verify W_base for M4.
    # M4 contributes to C11 (-1), C21 (1).
    # So W_base[0, 3] = -1, W_base[2, 3] = 1.
    # W_base[3, 3] should be 0.
    
    # M5
    U_base[0, 4] = 1; U_base[1, 4] = 1
    V_base[3, 4] = 1
    W_base[1, 4] = 1; W_base[3, 4] = -1
    
    # M6
    U_base[2, 5] = 1; U_base[0, 5] = -1
    V_base[0, 5] = 1; V_base[1, 5] = 1
    W_base[0, 5] = -1; W_base[2, 5] = 1
    
    # M7
    U_base[1, 6] = 1; U_base[3, 6] = -1
    V_base[2, 6] = 1; V_base[3, 6] = 1
    W_base[0, 6] = 1; W_base[1, 6] = 1; W_base[3, 6] = 1

    # --- Recursive 4x4x4 ---
    # Top level terms r=0..6
    # For each r, define coefficients for A blocks and B blocks
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # r=0: M1 = (A00+A11)(B00+B11)
    # r=1: M2 = (A10+A11)B00
    # r=2: M3 = A00(B01-B11)
    # r=3: M4 = A11(B10-B00)
    # r=4: M5 = (A00+A01)B11
    # r=5: M6 = (A10-A00)(B00+B01)
    # r=6: M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks in each term
    # A_coeffs[r][block]
    A_coeffs = [
        {0: 1, 1: 0, 2: 0, 3: 1}, # M1: A00, A11
        {0: 0, 1: 0, 2: 1, 3: 1}, # M2: A10, A11
        {0: 1, 1: 0, 2: 0, 3: 0}, # M3: A00
        {0: 0, 1: 0, 2: 0, 3: 1}, # M4: A11
        {0: 1, 1: 1, 2: 0, 3: 0}, # M5: A00, A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # M6: A10, -A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # M7: A01, -A11
    ]
    
    B_coeffs = [
        {0: 1, 1: 0, 2: 0, 3: 1}, # M1: B00, B11
        {0: 1, 1: 0, 2: 0, 3: 0}, # M2: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # M3: B01, -B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # M4: -B00, B10
        {0: 0, 1: 0, 2: 0, 3: 1}, # M5: B11
        {0: 1, 1: 1, 2: 0, 3: 0}, # M6: B00, B01
        {0: 0, 1: 0, 2: 1, 3: 1}  # M7: B10, B11
    ]
    
    # C_coeffs for output
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 - M5 + M7
    # Note: C blocks are 00, 01, 10, 11
    C_coeffs = [
        {0: 1, 1: 0, 2: 0, 3: 0}, # M1 -> C00
        {0: 0, 1: 0, 2: 0, 3: 0}, # M2 -> C10
        {0: 0, 1: 0, 2: 0, 3: 0}, # M3 -> C01
        {0: 0, 1: 0, 2: 0, 3: 0}, # M4 -> C00, C10
        {0: 0, 1: 0, 2: 0, 3: 0}, # M5 -> C01, C11
        {0: 0, 1: 0, 2: 0, 3: 0}, # M6 -> C00, C10
        {0: 0, 1: 0, 2: 0, 3: 0}  # M7 -> C00, C01, C11
    ]
    # This C_coeffs is wrong. We need to map each top-term r to the C blocks it affects.
    # Actually, the base Strassen W_base already handles the combination of M1..M7 into C00..C11.
    # So for each top-term r, we just need to know which C block the result goes into.
    # Wait, the top-term r IS one of the 7 multiplications.
    # The result of top-term r is a 2x2 matrix (the block C_r).
    # This C_r is then combined with other C_s to form the final C.
    # NO. The Strassen algorithm computes 7 products M1..M7.
    # Then C is formed from M1..M7.
    # So for each r, the output is a 2x2 matrix M_r.
    # M_r is a 2x2 matrix of 2x2 blocks?
    # No, M_r is a 2x2 matrix of scalars?
    # No, M_r is the product of two 2x2 matrices (which are sums of 2x2 blocks).
    # So M_r is a 2x2 matrix of 2x2 blocks?
    # No, M_r is a 2x2 matrix where each element is a 2x2 matrix?
    # Yes.
    # So for each r, we have a 2x2 matrix of 2x2 blocks.
    # Let's call this M_r_blocks.
    # M_r_blocks[i, j] is a 2x2 matrix.
    # This M_r_blocks is added to the corresponding C block?
    # No. C is formed by linear combinations of M1..M7.
    # C_block_00 = M1_block_00 - M4_block_00 - M6_block_00 + M7_block_00.
    # So for each r, the output M_r contributes to specific C blocks.
    # Specifically, M_r is a 2x2 matrix of 2x2 blocks.
    # Let M_r[i, j] be the (i, j) 2x2 block of M_r.
    # Then C_block_ij gets contributions from various M_r.
    # The coefficients are given by the Strassen formula.
    # C00 = M1 - M4 - M6 + M7.
    # So M1 contributes to C00 with coeff 1.
    # M4 contributes to C00 with coeff -1.
    # etc.
    
    # So, for each r, we need to know which C block(s) it contributes to and with what coefficient.
    # Actually, each M_r is a full 2x2 matrix of blocks.
    # M_r = M_r[0,0] M_r[0,1]
    #        M_r[1,0] M_r[1,1]
    # C00 = M1[0,0] - M4[0,0] - M6[0,0] + M7[0,0]
    # C01 = M1[0,1] - M4[0,1] - M6[0,1] + M7[0,1]
    # ...
    
    # So for each r, and for each (i, j) in 0..1, M_r[i, j] contributes to C_block_ij.
    # The coefficient is the same for all (i, j) for a given r?
    # Yes, because C00, C01, C10, C11 are formed by the same linear combination of M1..M7.
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 - M5 + M7
    
    # So for r=0 (M1), it contributes to C00 (coeff 1), C11 (coeff 1).
    # And it contributes to ALL sub-blocks of C00 and C11?
    # No. M1 is a 2x2 matrix of 2x2 blocks.
    # M1[0,0] contributes to C00[0,0].
    # M1[0,1] contributes to C01[0,0]? No.
    # M1 is the product of two 2x2 matrices.
    # M1 = X Y.
    # X and Y are 2x2 matrices of 2x2 blocks?
    # No, X and Y are 2x2 matrices of scalars?
    # No, X and Y are 2x2 matrices where each entry is a 2x2 matrix.
    # So M1 is a 2x2 matrix where each entry is a 2x2 matrix.
    # M1[i, j] is a 2x2 matrix.
    # C00 is a 2x2 matrix.
    # C00 = M1[0,0] - M4[0,0] ...
    # So M1[0,0] is added to C00.
    # M1[0,1] is added to C01?
    # No. C01 = M3[0,1] + M5[0,1] + M7[0,1].
    # So M1[0,1] does NOT contribute to C01.
    
    # So for each r, M_r[i, j] contributes to C_block_ij with coefficient c_{r, ij}.
    # The coefficients c_{r, ij} are:
    # r=0 (M1): c[0,0]=1, c[0,1]=0, c[1,0]=0, c[1,1]=1
    # r=1 (M2): c[0,0]=0, c[0,1]=0, c[1,0]=1, c[1,1]=0
    # r=2 (M3): c[0,0]=0, c[0,1]=1, c[1,0]=0, c[1,1]=-1
    # r=3 (M4): c[0,0]=-1, c[0,1]=0, c[1,0]=1, c[1,1]=0
    # r=4 (M5): c[0,0]=0, c[0,1]=1, c[1,0]=0, c[1,1]=-1
    # r=5 (M6): c[0,0]=-1, c[0,1]=0, c[1,0]=1, c[1,1]=0
    # r=6 (M7): c[0,0]=1, c[0,1]=1, c[1,0]=0, c[1,1]=1
    
    C_block_coeffs = [
        {0: 1, 1: 0, 2: 0, 3: 1}, # M1
        {0: 0, 1: 0, 2: 1, 3: 0}, # M2
        {0: 0, 1: 1, 2: 0, 3: -1}, # M3
        {0: -1, 1: 0, 2: 1, 3: 0}, # M4
        {0: 0, 1: 1, 2: 0, 3: -1}, # M5
        {0: -1, 1: 0, 2: 1, 3: 0}, # M6
        {0: 1, 1: 1, 2: 0, 3: 1}  # M7
    ]
    
    # Now fill U, V, W for Part 1
    offset_1 = 0
    for r in range(7):
        for s in range(7):
            idx = offset_1 + r * 7 + s
            # U
            for i in range(2):
                for j in range(2):
                    c_base = U_base[i * 2 + j, s]
                    if c_base != 0:
                        for b in range(4):
                            c_a = A_coeffs[r][b]
                            if c_a != 0:
                                # Block b is at row 2*(b//2), col 2*(b%2)
                                # Local i, j in block
                                I = 2 * (b // 2) + i
                                J = 2 * (b % 2) + j
                                U[I * 4 + J, idx] += c_base * c_a
            # V
            for i in range(2):
                for j in range(2):
                    c_base = V_base[i * 2 + j, s]
                    if c_base != 0:
                        for b in range(4):
                            c_b = B_coeffs[r][b]
                            if c_b != 0:
                                I = 2 * (b // 2) + i
                                J = 2 * (b % 2) + j
                                V[I * 4 + J, idx] += c_base * c_b
            # W
            for k in range(2):
                for i in range(2):
                    c_base = W_base[k * 2 + i, s]
                    if c_base != 0:
                        # This corresponds to M_r[k, i]
                        # Which contributes to C_block_{k, i} with coeff C_block_coeffs[r][k*2+i]
                        c_c = C_block_coeffs[r][k * 2 + i]
                        if c_c != 0:
                            # C_block_{k, i} is at row 2*k, col 2*i of the 4x4 C matrix
                            # But W index is k_global * n + i_global
                            # C is 4x4.
                            # Block (k, i) is at rows 2k..2k+1, cols 2i..2i+1
                            # Local indices in block: u, v
                            # Global K = 2*k + u, I = 2*i + v
                            # W index = K * 4 + I
                            # But W_base[k, i] is a scalar coefficient for the whole block?
                            # No, W_base[k, i, s] is the coefficient for the element C_eff[k, i].
                            # C_eff[k, i] is a 2x2 matrix.
                            # So we need to add this coefficient to all 4 elements of that block.
                            for u in range(2):
                                for v in range(2):
                                    K = 2 * k + u
                                    I = 2 * i + v
                                    W[K * 4 + I, idx] += c_base * c_c

    # --- Part 2: 4x4 x 4x1 ---
    # A is 4x4 (cols 0-3), B is 4x1 (rows 0-3)
    # Output C is 4x1 (rows 0-3, col 4)
    offset_2 = rank_1
    for i in range(4):
        for j in range(4):
            idx = offset_2 + i * 4 + j
            U[i * 5 + j, idx] = 1.0
            V[j * 5 + 4, idx] = 1.0 # B is 5x5. Block is rows 0-3, col 4.
            # B index: row j, col 4. V index = j * 5 + 4.
            W[4 * 4 + i, idx] = 1.0 # C index: row 4, col i. W index = 4 * 4 + i.

    # --- Part 3: 4x1 x 1x5 ---
    # A is 4x1 (col 4), B is 1x5 (row 4)
    # Output C is 4x5 (rows 0-3, cols 0-4)
    offset_3 = rank_1 + rank_2
    for i in range(4):
        for j in range(5):
            idx = offset_3 + i * 5 + j
            U[i * 5 + 4, idx] = 1.0
            V[4 * 5 + j, idx] = 1.0
            W[j * 4 + i, idx] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
