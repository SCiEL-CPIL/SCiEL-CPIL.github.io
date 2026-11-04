import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 using Recursive Strassen (Rank 49)
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: 4x4x4 Strassen ---
    
    # Define the 7 Strassen products for 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks: [A00, A01, A10, A11]
    # Coefficients for B blocks: [B00, B01, B10, B11]
    # Coefficients for C blocks: [C00, C01, C10, C11]
    
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    C_coeffs = [
        [1, 0, 0, 0], # C00
        [0, 1, 0, 0], # C01
        [0, 0, 1, 0], # C10
        [0, 0, 0, 1]  # C11
    ]
    
    # Reconstruction of C blocks from M's
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need the rank-7 decomposition of 2x2x2 multiplication.
    # Let's generate the U, V, W for 2x2x2 Strassen.
    # For 2x2x2, indices i,j,k in 0,1.
    # U22 (2x2, 7), V22 (2x2, 7), W22 (2x2, 7)
    
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # 2x2 Strassen terms
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We will fill U22, V22, W22 for these 7 terms.
    # Term 0: M1
    # A part: a00 + a11 -> U22[0,0]=1, U22[3,0]=1 (indices 0:00, 1:01, 2:10, 3:11)
    U22[0, 0] = 1; U22[3, 0] = 1
    # B part: b00 + b11 -> V22[0,0]=1, V22[3,0]=1
    V22[0, 0] = 1; V22[3, 0] = 1
    # C part: c00 + c01 + c10 + c11? No, M1 contributes to all.
    # C00 += M1, C01 += M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes to C00 (1), C11 (1).
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # Term 1: M2
    # A: a10 + a11 -> U22[2,1]=1, U22[3,1]=1
    U22[2, 1] = 1; U22[3, 1] = 1
    # B: b00 -> V22[0,1]=1
    V22[0, 1] = 1
    # C: C10 += M2, C11 -= M2
    W22[2, 1] = 1; W22[3, 1] = -1
    
    # Term 2: M3
    # A: a00 -> U22[0,2]=1
    U22[0, 2] = 1
    # B: b01 - b11 -> V22[1,2]=1, V22[3,2]=-1
    V22[1, 2] = 1; V22[3, 2] = -1
    # C: C01 += M3, C11 += M3
    W22[1, 2] = 1; W22[3, 2] = 1
    
    # Term 3: M4
    # A: a11 -> U22[3,3]=1
    U22[3, 3] = 1
    # B: b10 - b00 -> V22[2,3]=1, V22[0,3]=-1
    V22[2, 3] = 1; V22[0, 3] = -1
    # C: C00 += M4, C10 += M4
    W22[0, 3] = 1; W22[2, 3] = 1
    
    # Term 4: M5
    # A: a00 + a01 -> U22[0,4]=1, U22[1,4]=1
    U22[0, 4] = 1; U22[1, 4] = 1
    # B: b11 -> V22[3,4]=1
    V22[3, 4] = 1
    # C: C00 -= M5, C01 += M5
    W22[0, 4] = -1; W22[1, 4] = 1
    
    # Term 5: M6
    # A: a10 - a00 -> U22[2,5]=1, U22[0,5]=-1
    U22[2, 5] = 1; U22[0, 5] = -1
    # B: b00 + b01 -> V22[0,5]=1, V22[1,5]=1
    V22[0, 5] = 1; V22[1, 5] = 1
    # C: C11 += M6
    W22[3, 5] = 1
    
    # Term 6: M7
    # A: a01 - a11 -> U22[1,6]=1, U22[3,6]=-1
    U22[1, 6] = 1; U22[3, 6] = -1
    # B: b10 + b11 -> V22[2,6]=1, V22[3,6]=1
    V22[2, 6] = 1; V22[3, 6] = 1
    # C: C00 += M7
    W22[0, 6] = 1
    
    # Now construct U1, V1, W1 for 4x4x4
    # U1 (16, 49), V1 (16, 49), W1 (16, 49)
    U1 = np.zeros((16, 49))
    V1 = np.zeros((16, 49))
    W1 = np.zeros((16, 49))
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # A part
            for bi in range(4): # Block index 00, 01, 10, 11
                coeff = A_coeffs[s][bi]
                if coeff == 0: continue
                # Map block bi to local 2x2 indices
                # bi 0 -> (0,0), bi 1 -> (0,1), bi 2 -> (1,0), bi 3 -> (1,1)
                bi_i = bi // 2
                bi_j = bi % 2
                for li in range(2):
                    for lj in range(2):
                        global_i = bi_i * 2 + li
                        global_j = bi_j * 2 + lj
                        U1[global_i * 4 + global_j, col] += coeff * U22[li * 2 + lj, t]
                        
            # B part
            for bi in range(4):
                coeff = B_coeffs[s][bi]
                if coeff == 0: continue
                bi_j = bi // 2 # Block row index (j)
                bi_k = bi % 2 # Block col index (k)
                for lj in range(2):
                    for lk in range(2):
                        global_j = bi_j * 2 + lj
                        global_k = bi_k * 2 + lk
                        V1[global_j * 4 + global_k, col] += coeff * V22[lj * 2 + lk, t]
                        
            # C part
            # We need to sum contributions to C blocks.
            # C_block_index 0:00, 1:01, 2:10, 3:11
            # M_s contributes to C blocks with coefficients given by the reconstruction.
            # Let's define the contribution of M_s to C blocks.
            # M1: C00(1), C11(1)
            # M2: C10(1), C11(-1)
            # M3: C01(1), C11(1)
            # M4: C00(1), C10(1)
            # M5: C00(-1), C01(1)
            # M6: C11(1)
            # M7: C00(1)
            
            c_contrib = [0, 0, 0, 0]
            if s == 0: c_contrib = [1, 0, 0, 1]
            elif s == 1: c_contrib = [0, 0, 1, -1]
            elif s == 2: c_contrib = [0, 1, 0, 1]
            elif s == 3: c_contrib = [1, 0, 1, 0]
            elif s == 4: c_contrib = [-1, 1, 0, 0]
            elif s == 5: c_contrib = [0, 0, 0, 1]
            elif s == 6: c_contrib = [1, 0, 0, 0]
            
            for bi in range(4):
                coeff = c_contrib[bi]
                if coeff == 0: continue
                bi_i = bi // 2 # Block row index (i)
                bi_k = bi % 2 # Block col index (k)
                for li in range(2):
                    for lk in range(2):
                        global_i = bi_i * 2 + li
                        global_k = bi_k * 2 + lk
                        W1[global_k * 4 + global_i, col] += coeff * W22[li * 2 + lk, t]

    # --- Part 2: 4x4x3 Schoolbook ---
    U2 = np.zeros((16, 48))
    V2 = np.zeros((12, 48))
    W2 = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2[i * 4 + j, col] = 1
                V2[j * 3 + k, col] = 1
                W2[k * 4 + i, col] = 1
                col += 1

    # --- Assemble Full U, V, W ---
    
    # U
    U[:, 0:49] = U1
    U[:, 49:97] = U2
    
    # V
    # V1 (16, 49) maps to V[0:16, 0:49]
    # But V is (28, 97). Rows 0-15 are B_left (cols 0-3).
    # V1 row index j*4+k. Full V row index j*7+k.
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, 0:49] = V1[j * 4 + k, 0:49]
            
    # V2 (12, 48) maps to V[16:28, 49:97]
    # V2 row index j*3+k. Full V row index j*7 + (k+4).
    for j in range(4):
        for k in range(3):
            V[j * 7 + k + 4, 49:97] = V2[j * 3 + k, 0:48]
            
    # W
    # W1 (16, 49) maps to W[0:16, 0:49]
    # W1 row index k*4+i. Full W row index k*4+i.
    W[0:16, 0:49] = W1
    
    # W2 (12, 48) maps to W[16:28, 49:97]
    # W2 row index k*4+i. Full W row index (k+4)*4 + i.
    for k in range(3):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:97] = W2[k * 4 + i, 0:48]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
