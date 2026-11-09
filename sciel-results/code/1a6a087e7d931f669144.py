"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate 2x2x2 Strassen decomposition
    # U2, V2, W2 of shape (4, 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen terms for 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indices for 2x2:
    # A: 0(11), 1(12), 2(21), 3(22) -> i*2+j
    # B: 0(11), 1(12), 2(21), 3(22) -> j*2+k
    # C: 0(11), 1(12), 2(21), 3(22) -> k*2+i
    
    # Term 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = -1
    
    # Term 2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # Term 3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[0, 3] = 1
    
    # Term 5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = -1
    
    # Term 6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[2, 6] = 1
    
    # 2. Construct 4x4x4 decomposition using 2x2x2 blocks
    # Rank 49
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49)) # Will be part of final W (rows 0-15)
    
    # Define Strassen steps for 4x4
    # Blocks A_R,C and B_C,K
    # Step s:
    # X_s = sum c_R,C A_R,C
    # Y_s = sum d_C,K B_C,K
    # M_s = X_s Y_s
    # C_R,K = sum w_s_R,K M_s
    
    # Coefficients c_s[R,C], d_s[C,K], w_s[R,K]
    # R,C,K in {0,1}
    
    steps = [
        # s, c_dict, d_dict, w_dict
        (0, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):-1}),
        (1, {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        (2, {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        (3, {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        (4, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        (5, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        (6, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):1})
    ]
    
    # Wait, let's verify the W coefficients from standard Strassen
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My w_dict above:
    # s=0 (M1): (0,0):1, (0,1):1, (1,0):1, (1,1):-1 -> WRONG.
    # C11 gets +1, C12 gets +1? No.
    # M1 contributes to C11 (+1), C12 (+1)? No.
    # C11 = M1 + ...
    # C12 = M3 + M5 ... M1 is not in C12.
    # C21 = M2 + M4 ... M1 is not in C21.
    # C22 = M1 - M2 + ...
    # So M1 contributes to C11 (+1) and C22 (+1).
    # Let's re-verify Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Correct w_dict:
    steps_corrected = [
        # s, c_dict, d_dict, w_dict
        (0, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        (1, {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        (2, {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        (3, {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        (4, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        (5, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        (6, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):1})  # M7
    ]
    
    for s, c_dict, d_dict, w_dict in steps_corrected:
        for t in range(7):
            col = s * 7 + t
            # Fill U444
            for R in range(2):
                for C in range(2):
                    c_val = c_dict.get((R,C), 0)
                    if c_val != 0:
                        # A_R,C block indices: i in 0..1, j in 0..1
                        # Global A index: (2R+i)*4 + (2C+j) ? No.
                        # A is 4x4. Row 2R+i, Col 2C+j.
                        # Index in U: (2R+i)*m + (2C+j) = (2R+i)*4 + (2C+j)
                        for i in range(2):
                            for j in range(2):
                                idx_u = (2*R + i) * 4 + (2*C + j)
                                # U2 index: i*2 + j
                                idx_u2 = i * 2 + j
                                U444[idx_u, col] += c_val * U2[idx_u2, t]
                                
            # Fill V444
            for C in range(2):
                for K in range(2):
                    d_val = d_dict.get((C,K), 0)
                    if d_val != 0:
                        # B_C,K block indices: j in 0..1, k in 0..1
                        # Global B index: (2C+j)*4 + (2K+k) ?
                        # B is 4x4. Row 2C+j, Col 2K+k.
                        # Index in V: (2C+j)*p + (2K+k) = (2C+j)*4 + (2K+k)
                        for j in range(2):
                            for k in range(2):
                                idx_v = (2*C + j) * 4 + (2*K + k)
                                # V2 index: j*2 + k
                                idx_v2 = j * 2 + k
                                V444[idx_v, col] += d_val * V2[idx_v2, t]
                                
            # Fill W444 (for 4x4x4 part, rows 0-15)
            for R in range(2):
                for K in range(2):
                    w_val = w_dict.get((R,K), 0)
                    if w_val != 0:
                        # C_R,K block indices: i in 0..1, k in 0..1
                        # Global C index: k_global * n + i_global
                        # k_global = 2K+k, i_global = 2R+i
                        # Index in W: (2K+k)*4 + (2R+i)
                        for i in range(2):
                            for k in range(2):
                                idx_w = (2*K + k) * 4 + (2*R + i)
                                # W2 index: k*2 + i
                                idx_w2 = k * 2 + i
                                W444[idx_w, col] += w_val * W2[idx_w2, t]

    # 3. Add 4x4x1 part
    # Rank 16
    # U (16, 16), V (4, 16), W (4, 16)
    # Append to U444, V444, and new W rows
    
    U_final = np.zeros((16, 49 + 16))
    V_final = np.zeros((16, 49 + 16)) # B is 4x5, so V is (4*5, rank) = (20, rank)
    # Wait. V shape is (m*p, rank). m=4, p=5. So V is (20, rank).
    # My V444 was (16, 49). It corresponds to B_left (4x4).
    # We need to append V_right (4x1) part.
    
    W_final = np.zeros((20, 49 + 16)) # C is 5x4. W is (5*4, rank) = (20, rank).
    
    # Copy 4x4x4 part
    U_final[:, :49] = U444
    V_final[:16, :49] = V444 # B_left rows 0-15
    W_final[:16, :49] = W444 # C_left rows 0-15 (k=0..3)
    
    # Fill 4x4x1 part
    # A (4x4), B_right (4x1), C_right (1x4)
    # B_right indices: j in 0..3, k=4.
    # Global B index: j*5 + 4.
    # C_right indices: k=4, i in 0..3.
    # Global C index: 4*4 + i = 16 + i.
    
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            # U
            U_final[i * 4 + j, col] = 1
            # V
            V_final[j * 5 + 4, col] = 1
            # W
            W_final[16 + i, col] = 1

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
