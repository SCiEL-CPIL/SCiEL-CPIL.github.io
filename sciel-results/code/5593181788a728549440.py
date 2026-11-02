import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Global sizes
    U_size = n * m # 16
    V_size = m * p # 24
    W_size = p * n # 24
    
    # We will build the decomposition term by term
    # Total terms expected: 7 (Strassen blocks) * 11 (Sub-decomp) = 77
    
    global_U = []
    global_V = []
    global_W = []
    
    # --- Step 1: Define Sub-Decomposition for 2x2x3 (Rank 11) ---
    
    # 1a. Strassen 2x2x2 (Rank 7)
    # U, V are 2x2. W is 2x2.
    strassen_U = []
    strassen_V = []
    strassen_W = []
    
    # M1 = (A00+A11)(B00+B11)
    strassen_U.append([[1, 0], [1, 0]]) # A00 + A11? No, U is for A.
    # Wait, U_s^T A.
    # If U_s = [[1,0],[1,0]], U_s^T A = [[1,1],[0,0]] dot A = A00+A10?
    # Indices: A = [[A00, A01], [A10, A11]]? No, A is 2x2.
    # A = [[a00, a01], [a10, a11]].
    # U_s = [[1, 0], [1, 0]] -> U_s^T = [[1, 1], [0, 0]].
    # U_s^T A = [[a00+a10, a01+a11], [0,0]].
    # This is not (A00+A11).
    # Strassen M1: (A00+A11).
    # We need U_s such that U_s^T A = A00+A11 (as a scalar? No, A is 2x2).
    # Wait, in the sub-problem, A is 2x2.
    # Strassen M1 = (A00+A11)(B00+B11).
    # Here A00 is the top-left element of the 2x2 matrix A.
    # So A00 is a scalar.
    # U_s must extract the linear combination of A's entries.
    # U_s is 2x2. U_s^T A is a scalar?
    # No, U_s^T A is a 2x2 matrix?
    # Let's re-verify the tensor definition.
    # T = sum U_r o V_r o W_r.
    # C = sum (U_r^T A) (V_r^T B) W_r^T.
    # If A is 2x2, U_r is 2x2. U_r^T A is 2x2.
    # Then (U_r^T A) (V_r^T B) is 2x2 dot 2x2?
    # V_r is 2x2. V_r^T B is 2x2.
    # So (U_r^T A) (V_r^T B) is 2x2.
    # Then dot W_r^T (2x2) is 2x2.
    # This works for 2x2x2.
    
    # So for Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11).
    # We need U_1 such that U_1^T A = A00+A11 (scalar? No).
    # Wait, the product is (Scalar) * (Scalar) * Matrix?
    # No.
    # In the standard decomposition, U_r, V_r, W_r are matrices.
    # The term is (U_r : A) * (V_r : B) * W_r.
    # Where : is the Frobenius inner product (trace of transpose product).
    # So (U_r : A) is a scalar.
    # So U_r is 2x2.
    
    # Strassen M1: (A00+A11)(B00+B11).
    # We need U_1 such that U_1 : A = A00+A11.
    # U_1 = [[1, 0], [0, 1]].
    # V_1 = [[1, 0], [0, 1]].
    # W_1 = I (2x2 identity).
    
    # M2 = (A10+A11)B00.
    # U_2 = [[0, 0], [1, 1]].
    # V_2 = [[1, 0], [0, 0]].
    # W_2 = [[0, 0], [1, 0]]?
    # C = (U:A)(V:B)W.
    # (U2:A) = A10+A11.
    # (V2:B) = B00.
    # W2 should place this at C10 (bottom-left).
    # C = [[0,0],[1,0]] * scalar?
    # C10 = scalar.
    # So W2 = [[0,0],[1,0]].
    
    # M3 = A00(B01-B11).
    # U3 = [[1,0],[0,0]].
    # V3 = [[0,1],[0,-1]].
    # W3 = [[0,1],[0,0]]?
    # C01 = scalar.
    # W3 = [[0,1],[0,0]].
    
    # M4 = A11(B10-B00).
    # U4 = [[0,0],[0,1]].
    # V4 = [[-1,0],[1,0]].
    # W4 = [[0,0],[1,0]].
    
    # M5 = (A00+A01)B11.
    # U5 = [[1,1],[0,0]].
    # V5 = [[0,0],[0,1]].
    # W5 = [[0,1],[0,0]].
    
    # M6 = (A10-A00)(B00+B01).
    # U6 = [[-1,0],[1,0]].
    # V6 = [[1,1],[0,0]].
    # W6 = [[1,0],[0,0]].
    
    # M7 = (A01-A11)(B10+B11).
    # U7 = [[0,1],[-1,0]].
    # V7 = [[0,0],[1,1]].
    # W7 = [[1,0],[0,0]].
    
    # Check reconstruction:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # This is correct.
    
    strassen_U = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[-1,0]]
    ]
    strassen_V = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ]
    strassen_W = [
        [[1,0],[0,1]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[1,0],[0,0]],
        [[1,0],[0,0]]
    ]
    
    # 1b. Schoolbook 2x2x1 (Rank 4)
    # A is 2x2. B is 2x1. C is 2x1.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    school_U = [
        [[1,0],[0,0]], # A00
        [[0,1],[0,0]], # A01
        [[0,0],[1,0]], # A10
        [[0,0],[0,1]]  # A11
    ]
    school_V = [
        [[1],[0]], # B0
        [[1],[0]], # B0
        [[1],[0]], # B0
        [[1],[0]]  # B0 -- Wait
        # C0 = A00 B0 + A01 B1.
        # Term 1: A00 * B0. U=[[1,0],[0,0]], V=[[1],[0]], W=[[1],[0]].
        # Term 2: A01 * B1. U=[[0,1],[0,0]], V=[[0],[1]], W=[[1],[0]].
        # Term 3: A10 * B0. U=[[0,0],[1,0]], V=[[1],[0]], W=[[0],[1]].
        # Term 4: A11 * B1. U=[[0,0],[0,1]], V=[[0],[1]], W=[[0],[1]].
    ]
    school_V = [
        [[1],[0]],
        [[0],[1]],
        [[1],[0]],
        [[0],[1]]
    ]
    school_W = [
        [[1],[0]],
        [[1],[0]],
        [[0],[1]],
        [[0],[1]]
    ]
    
    # 1c. Combine into 2x2x3 Sub-Decomposition (Rank 11)
    # U_s (2x2), V_s (2x3), W_s (2x3)
    
    sub_U = []
    sub_V = []
    sub_W = []
    
    # Strassen part (cols 0,1 of output)
    for i in range(7):
        U_s = strassen_U[i]
        V_s_2x2 = strassen_V[i]
        W_s_2x2 = strassen_W[i]
        
        # Pad V to 2x3
        V_s = [
            [V_s_2x2[0][0], V_s_2x2[0][1], 0],
            [V_s_2x2[1][0], V_s_2x2[1][1], 0]
        ]
        # Pad W to 2x3
        W_s = [
            [W_s_2x2[0][0], W_s_2x2[0][1], 0],
            [W_s_2x2[1][0], W_s_2x2[1][1], 0]
        ]
        sub_U.append(U_s)
        sub_V.append(V_s)
        sub_W.append(W_s)
        
    # Schoolbook part (col 2 of output)
    for i in range(4):
        U_s = school_U[i]
        V_s_2x1 = school_V[i]
        W_s_2x1 = school_W[i]
        
        # Pad V to 2x3 (col 2)
        V_s = [
            [0, 0, V_s_2x1[0][0]],
            [0, 0, V_s_2x1[1][0]]
        ]
        # Pad W to 2x3 (col 2)
        W_s = [
            [0, 0, W_s_2x1[0][0]],
            [0, 0, W_s_2x1[1][0]]
        ]
        sub_U.append(U_s)
        sub_V.append(V_s)
        sub_W.append(W_s)
        
    # --- Step 2: Global Strassen Blocks ---
    
    # Global A blocks: A00, A01, A10, A11 (each 2x2)
    # Global B blocks: B00, B01, B10, B11 (each 2x3)
    
    # Strassen terms for global 4x4x6
    # X_t (2x2), Y_t (2x3)
    
    global_strassen_X = [] # Coeffs for A blocks
    global_strassen_Y = [] # Coeffs for B blocks
    global_strassen_alpha = [] # Coeffs for C blocks (C00, C01, C10, C11)
    
    # M1 = (A00+A11)(B00+B11)
    global_strassen_X.append([[1,0,0,0], [0,0,0,1]]) # A00, A01, A10, A11
    global_strassen_Y.append([[1,0,0,0], [0,0,1,0]]) # B00, B01, B10, B11
    global_strassen_alpha.append([[1,0], [0,1]]) # C00, C01, C10, C11?
    # Alpha is 2x2 matrix of coefficients for C blocks.
    # C00 += 1*M1, C01 += 0, C10 += 0, C11 += 1?
    # Reconstruction:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # Let's store alpha as 4x4? No, 2x2 blocks.
    # alpha[t][block_row][block_col]
    
    alpha_1 = [[1, 0], [0, 1]] # M1 contributes to C00 and C11?
    # Wait.
    # C00 = M1 - M4 + M6 + M7
    # C11 = M1 - M3 + M2 - M5
    # So M1 contributes 1 to C00 and 1 to C11.
    # M4 contributes -1 to C00 and 1 to C10.
    # M6 contributes 1 to C00.
    # M7 contributes 1 to C00.
    # M3 contributes 1 to C01 and -1 to C11.
    # M5 contributes 1 to C01 and -1 to C11.
    # M2 contributes 1 to C10.
    
    alphas = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [0, 1]], # M4
        [[0, 1], [0, -1]], # M5
        [[1, 0], [0, 0]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    # X and Y definitions
    # X is 2x2 matrix of coeffs for A00, A01, A10, A11?
    # No, X is 2x2 matrix. Each entry is a linear combo of A blocks.
    # Actually, X_t is a 2x2 matrix.
    # X_t = sum c_{ij} A_{ij}?
    # No, X_t is formed by combining A blocks.
    # M1: X = A00+A11. This is a 2x2 matrix.
    # A00 is top-left 2x2 of A. A11 is bottom-right 2x2.
    # So X = [[A00, 0], [0, A11]]? No.
    # A = [[A00, A01], [A10, A11]].
    # A00+A11 is not a block matrix. It's the sum of the two 2x2 blocks.
    # So X = A00 + A11.
    
    # We need to represent X as a linear combination of the 4 A blocks.
    # X = c00 A00 + c01 A01 + c10 A10 + c11 A11.
    # For M1: c00=1, c11=1, others 0.
    
    X_coeffs = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]], # M6: A10 - A00
        [[0, 1], [-1, 0]]  # M7: A01 - A11
    ]
    
    Y_coeffs = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]], # M3: B01 - B11
        [[-1, 0], [1, 0]], # M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 0], [1, 1]]  # M7: B10 + B11
    ]
    
    # --- Step 3: Construct Global U, V, W ---
    
    rank = 7 * 11
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    col = 0
    for t in range(7):
        X_c = X_coeffs[t] # 2x2
        Y_c = Y_coeffs[t] # 2x2
        alpha = alphas[t] # 2x2
        
        for s in range(11):
            U_s = sub_U[s] # 2x2
            V_s = sub_V[s] # 2x3
            W_s = sub_W[s] # 2x3
            
            # Calculate U_global contribution
            # U_s^T X_t is a scalar?
            # No.
            # X_t is a 2x2 matrix of coefficients for A blocks.
            # This is tricky.
            # X_t = sum_{i,j} X_c[i][j] A_{ij}.
            # U_s^T X_t = sum_{i,j} X_c[i][j] (U_s^T A_{ij}).
            # U_s^T A_{ij} is a linear combination of A_{ij}'s entries.
            # We need to add this to U_global.
            
            # U_global row corresponds to A entry.
            # A_{ij} is 2x2.
            # A_{00} entries: (0,0), (0,1), (1,0), (1,1) -> Global indices 0,1,2,3?
            # A is 4x4.
            # A00: rows 0,1 cols 0,1.
            # A01: rows 0,1 cols 2,3.
            # A10: rows 2,3 cols 0,1.
            # A11: rows 2,3 cols 2,3.
            
            # For each block (bi, bj) in 0..1:
            for bi in range(2):
                for bj in range(2):
                    coeff = X_c[bi][bj]
                    if coeff == 0: continue
                    
                    # Block A_{bi,bj}
                    # U_s is 2x2.
                    # U_s^T A_{bi,bj} is a scalar.
                    # (U_s^T A)_{kl} = sum_{pq} U_{pq} A_{kp, lq}?
                    # Frobenius inner product: sum_{pq} U_{pq} A_{pq}.
                    # So we add coeff * U_s[p][q] to U_global[global_idx][col].
                    
                    # Global index for A_{bi,bj}[p][q]
                    # Row: bi*2 + p
                    # Col: bj*2 + q
                    # Index: (bi*2+p)*4 + (bj*2+q)
                    
                    for p in range(2):
                        for q in range(2):
                            g_idx = (bi*2 + p)*4 + (bj*2 + q)
                            U[g_idx][col] += coeff * U_s[p][q]
                            
            # Calculate V_global contribution
            # V_s^T Y_t
            # Y_t = sum_{bi,bj} Y_c[bi][bj] B_{bi,bj}.
            # V_s is 2x3.
            # B_{bi,bj} is 2x3.
            # Inner product: sum V_s[p][k] B_{bi,bj}[p][k].
            
            for bi in range(2):
                for bj in range(2):
                    coeff = Y_c[bi][bj]
                    if coeff == 0: continue
                    
                    # Block B_{bi,bj}
                    # Global index for B_{bi,bj}[p][k]
                    # Row: bi*2 + p
                    # Col: bj*3 + k
                    # Index: (bi*2+p)*6 + (bj*3+k)
                    
                    for p in range(2):
                        for k in range(3):
                            g_idx = (bi*2 + p)*6 + (bj*3 + k)
                            V[g_idx][col] += coeff * V_s[p][k]
                            
            # Calculate W_global contribution
            # alpha[bi][bj] * W_s
            # W_s is 2x3.
            # Output block C_{bi,bj} is 2x3.
            # Global index for C_{bi,bj}[i][k]
            # Row: bi*2 + i
            # Col: bj*3 + k
            # Index: (bj*3+k)*4 + (bi*2+i)  # W[k*n+i]
            
            for bi in range(2):
                for bj in range(2):
                    coeff = alpha[bi][bj]
                    if coeff == 0: continue
                    
                    for i in range(2):
                        for k in range(3):
                            g_idx = (bj*3 + k)*4 + (bi*2 + i)
                            W[g_idx][col] += coeff * W_s[i][k]
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}

