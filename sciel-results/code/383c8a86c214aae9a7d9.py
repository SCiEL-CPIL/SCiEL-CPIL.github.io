import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # Rank 49 for 4x4x4 (Strassen) + Rank 4 for 4x4x1 (Schoolbook) = 53
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 16 x 53
    V = [[0] * rank for _ in range(m * p)]  # 20 x 53
    W = [[0] * rank for _ in range(p * n)]  # 20 x 53
    
    # --- Part 1: 4x4x4 Multiplication (Rank 49) ---
    # We use Strassen's algorithm for 4x4 matrices.
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 rows of C? No, C is 5x4).
    # C[i, k] for k=0..3 depends on A[i, j] and B[j, k].
    
    # Strassen for 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    
    # The 7 Strassen products M1..M7 are 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7.
    # So we need 7 * 7 = 49 terms.
    
    # Define the 2x2x2 Strassen decomposition (Rank 7)
    # For matrices X, Y (2x2), Z = X Y.
    # Z[0,0] = M1 + M4 - M5 + M7
    # Z[0,1] = M3 + M5
    # Z[1,0] = M2 + M4
    # Z[1,1] = M1 - M2 + M3 + M6
    # Where:
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # We need to map these to the global indices.
    # Global A indices: i, j in 0..3.
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2,3
    # A21: i in 2,3; j in 0,1
    # A22: i in 2,3; j in 2,3
    
    # Global B indices: j, k in 0..3 (for the 4x4 part).
    # B11: j in 0,1; k in 0,1
    # B12: j in 0,1; k in 2,3
    # B21: j in 2,3; k in 0,1
    # B22: j in 2,3; k in 2,3
    
    # Global C indices: k, i in 0..3 (for the 4x4 part, k=0..3, i=0..3).
    # Note: W is indexed by [k*n + i]. So W[k*4 + i].
    
    # For each of the 7 Strassen steps (s=0..6), we have a 2x2 multiplication.
    # Let s be the step index.
    # The X matrix for step s is a linear combination of A blocks.
    # The Y matrix for step s is a linear combination of B blocks.
    
    # Coefficients for X (A blocks) and Y (B blocks) for each step s.
    # X = a11*A11 + a12*A12 + a21*A21 + a22*A22
    # Y = b11*B11 + b12*B12 + b21*B21 + b22*B22
    
    # s=0 (M1): X = A11 + A22, Y = B11 + B22
    # s=1 (M2): X = A21 + A22, Y = B11
    # s=2 (M3): X = A11, Y = B12 - B22
    # s=3 (M4): X = A22, Y = B21 - B11
    # s=4 (M5): X = A11 + A12, Y = B22
    # s=5 (M6): X = A21 - A11, Y = B11 + B12
    # s=6 (M7): X = A12 - A22, Y = B21 + B22
    
    # For each step s, we have a 2x2x2 multiplication.
    # The 2x2x2 multiplication has 7 terms (t=0..6).
    # Let's define the 2x2x2 Strassen decomposition coefficients.
    # For a generic 2x2 multiplication Z = X Y:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # The 7 products are:
    # P0 = (x00 + x11)(y00 + y11)
    # P1 = (x10 + x11)y00
    # P2 = x00(y01 - y11)
    # P3 = x11(y10 - y00)
    # P4 = (x00 + x01)y11
    # P5 = (x10 - x00)(y00 + y01)
    # P6 = (x01 - x11)(y10 + y11)
    
    # Z00 = P0 + P3 - P4 + P6
    # Z01 = P2 + P4
    # Z10 = P1 + P3
    # Z11 = P0 - P1 + P2 + P5
    
    # We will iterate over s (0..6) and t (0..6).
    # Term index in global U, V, W: idx = s * 7 + t.
    
    for s in range(7):
        # Determine coefficients for X (A blocks)
        # a11, a12, a21, a22
        if s == 0:
            a11, a12, a21, a22 = 1, 0, 0, 1
        elif s == 1:
            a11, a12, a21, a22 = 0, 0, 1, 1
        elif s == 2:
            a11, a12, a21, a22 = 1, 0, 0, 0
        elif s == 3:
            a11, a12, a21, a22 = 0, 0, 0, 1
        elif s == 4:
            a11, a12, a21, a22 = 1, 1, 0, 0
        elif s == 5:
            a11, a12, a21, a22 = -1, 0, 1, 0
        elif s == 6:
            a11, a12, a21, a22 = 0, 1, 0, -1
        
        # Determine coefficients for Y (B blocks)
        # b11, b12, b21, b22
        if s == 0:
            b11, b12, b21, b22 = 1, 0, 0, 1
        elif s == 1:
            b11, b12, b21, b22 = 1, 0, 0, 0
        elif s == 2:
            b11, b12, b21, b22 = 0, 1, 0, -1
        elif s == 3:
            b11, b12, b21, b22 = 0, 0, 1, -1
        elif s == 4:
            b11, b12, b21, b22 = 0, 0, 0, 1
        elif s == 5:
            b11, b12, b21, b22 = 1, 1, 0, 0
        elif s == 6:
            b11, b12, b21, b22 = 0, 0, 1, 1
            
        for t in range(7):
            idx = s * 7 + t
            
            # Determine coefficients for x (entries of X) and y (entries of Y)
            # X is a 2x2 matrix. x00, x01, x10, x11 are linear combos of A blocks.
            # But X is formed by combining A blocks.
            # X[i, j] = sum_{block} coeff_block * A_block[i, j]
            # For a specific entry x_ij of X, it is a linear combination of A entries.
            # Similarly for y_kl.
            
            # The 7 products P0..P6 for 2x2x2 are defined by linear forms of x and y.
            # Let's define the linear forms for x and y for each t.
            
            # P0: x00+x11, y00+y11
            # P1: x10+x11, y00
            # P2: x00, y01-y11
            # P3: x11, y10-y00
            # P4: x00+x01, y11
            # P5: x10-x00, y00+y01
            # P6: x01-x11, y10+y11
            
            # We need to map x_ij to A entries and y_kl to B entries.
            # x_ij corresponds to the (i,j) entry of the 2x2 matrix X.
            # X is composed of A blocks.
            # X[0,0] is the top-left entry of the 2x2 matrix X.
            # This entry is a linear combination of A11[0,0], A12[0,0], A21[0,0], A22[0,0].
            # Wait, X is a 2x2 matrix of scalars? No, X is a 2x2 matrix of 2x2 blocks?
            # No, in Strassen, X is a 2x2 matrix of scalars?
            # No, Strassen works on blocks.
            # M1 = (A11 + A22)(B11 + B22).
            # Here A11+A22 is a 2x2 matrix. B11+B22 is a 2x2 matrix.
            # Their product is a 2x2 matrix.
            # So X is a 2x2 matrix of scalars (formed by adding blocks).
            # So x00 is the (0,0) entry of the matrix (A11 + A22).
            # x00 = A11[0,0] + A22[0,0].
            # A11[0,0] is A[0,0]. A22[0,0] is A[2,2].
            # So x00 = A[0,0] + A[2,2].
            
            # Let's define the mapping from X entries to A indices.
            # X[0,0] = a11*A[0,0] + a12*A[0,2] + a21*A[2,0] + a22*A[2,2]
            # X[0,1] = a11*A[0,1] + a12*A[0,3] + a21*A[2,1] + a22*A[2,3]
            # X[1,0] = a11*A[1,0] + a12*A[1,2] + a21*A[3,0] + a22*A[3,2]
            # X[1,1] = a11*A[1,1] + a12*A[1,3] + a21*A[3,1] + a22*A[3,3]
            
            # Similarly for Y entries to B indices.
            # Y[0,0] = b11*B[0,0] + b12*B[0,2] + b21*B[2,0] + b22*B[2,2]
            # Y[0,1] = b11*B[0,1] + b12*B[0,3] + b21*B[2,1] + b22*B[2,3]
            # Y[1,0] = b11*B[1,0] + b12*B[1,2] + b21*B[3,0] + b22*B[3,2]
            # Y[1,1] = b11*B[1,1] + b12*B[1,3] + b21*B[3,1] + b22*B[3,3]
            
            # Now, for each t (0..6), we have linear forms of x and y.
            # We need to expand these to get coefficients for A and B entries.
            
            # Let's define the coefficients for x (0..3) and y (0..3) for each t.
            # x_coeffs = [c00, c01, c10, c11]
            # y_coeffs = [c00, c01, c10, c11]
            
            if t == 0:
                x_c = [1, 0, 0, 1]  # x00 + x11
                y_c = [1, 0, 0, 1]  # y00 + y11
            elif t == 1:
                x_c = [0, 0, 1, 1]  # x10 + x11
                y_c = [1, 0, 0, 0]  # y00
            elif t == 2:
                x_c = [1, 0, 0, 0]  # x00
                y_c = [0, 1, 0, -1] # y01 - y11
            elif t == 3:
                x_c = [0, 0, 0, 1]  # x11
                y_c = [0, 0, 1, -1] # y10 - y00
            elif t == 4:
                x_c = [1, 1, 0, 0]  # x00 + x01
                y_c = [0, 0, 0, 1]  # y11
            elif t == 5:
                x_c = [0, 0, 1, -1] # x10 - x00
                y_c = [1, 1, 0, 0]  # y00 + y01
            elif t == 6:
                x_c = [0, 1, 0, -1] # x01 - x11
                y_c = [0, 0, 1, 1]  # y10 + y11
            
            # Now, for each A entry (i, j), calculate the coefficient in U.
            # U[i*4 + j, idx]
            for i in range(4):
                for j in range(4):
                    # Determine which block A[i,j] belongs to
                    # Block 11: i in 0,1; j in 0,1
                    # Block 12: i in 0,1; j in 2,3
                    # Block 21: i in 2,3; j in 0,1
                    # Block 22: i in 2,3; j in 2,3
                    
                    # Local indices in the 2x2 block
                    li = i % 2
                    lj = j % 2
                    
                    # Which block?
                    if i < 2 and j < 2:
                        block = 0 # 11
                        coeff_block = a11
                    elif i < 2 and j >= 2:
                        block = 1 # 12
                        coeff_block = a12
                    elif i >= 2 and j < 2:
                        block = 2 # 21
                        coeff_block = a21
                    else:
                        block = 3 # 22
                        coeff_block = a22
                    
                    # The entry A[i,j] contributes to X[li, lj].
                    # The coefficient of X[li, lj] in the linear form for t is x_c[block_in_x]
                    # Wait, x_c is [x00, x01, x10, x11].
                    # x00 corresponds to li=0, lj=0.
                    # x01 corresponds to li=0, lj=1.
                    # x10 corresponds to li=1, lj=0.
                    # x11 corresponds to li=1, lj=1.
                    
                    x_idx = li * 2 + lj
                    coeff_x = x_c[x_idx]
                    
                    U[i * 4 + j][idx] = coeff_block * coeff_x
            
            # Similarly for V
            for j in range(4):
                for k in range(4):
                    li = j % 2
                    lj = k % 2
                    
                    if j < 2 and k < 2:
                        coeff_block = b11
                    elif j < 2 and k >= 2:
                        coeff_block = b12
                    elif j >= 2 and k < 2:
                        coeff_block = b21
                    else:
                        coeff_block = b22
                    
                    y_idx = li * 2 + lj
                    coeff_y = y_c[y_idx]
                    
                    V[j * 5 + k][idx] = coeff_block * coeff_y
            
            # Similarly for W
            # C = Z.
            # Z is a 2x2 matrix of 2x2 blocks?
            # No, Z is the result of the 2x2 multiplication of X and Y.
            # Z is a 2x2 matrix of scalars?
            # No, X and Y are 2x2 matrices of scalars (formed from blocks).
            # So Z is a 2x2 matrix of scalars.
            # But C is a 4x4 matrix.
            # The Strassen algorithm computes the 4 block entries C11, C12, C21, C22.
            # Each C_block is a 2x2 matrix.
            # So for each of the 4 block positions in C, we have a 2x2 matrix of entries.
            # The Strassen formulas give the values for the 4 blocks.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Here M1..M7 are the 7 products from the current step s?
            # No, M1..M7 are the 7 products for the *entire* 4x4 multiplication.
            # In my loop, s is the step index (0..6) corresponding to M1..M7.
            # t is the term index within the 2x2 multiplication for that step.
            # So for a fixed s, we are computing the contribution of M_s to C.
            # M_s is a 2x2 matrix (result of X_s * Y_s).
            # The entries of M_s are linear combinations of the 7 products P0..P6.
            # Let's denote the 7 products as P_t.
            # M_s[0,0] = P0 + P3 - P4 + P6
            # M_s[0,1] = P2 + P4
            # M_s[1,0] = P1 + P3
            # M_s[1,1] = P0 - P1 + P2 + P5
            
            # Now, how does M_s contribute to C?
            # C11 += M1 (if s=0)
            # C12 += M3 (if s=2) + M5 (if s=4)
            # C21 += M2 (if s=1) + M4 (if s=3)
            # C22 += M1 (if s=0) - M2 (if s=1) + M3 (if s=2) + M6 (if s=5)
            
            # So for a given s, we know which C blocks are affected and with what sign.
            
            c_blocks = {}
            if s == 0: # M1
                c_blocks = {0: 1, 3: 1} # C11 += M1, C22 += M1
            elif s == 1: # M2
                c_blocks = {2: 1, 3: -1} # C21 += M2, C22 -= M2
            elif s == 2: # M3
                c_blocks = {1: 1, 3: 1} # C12 += M3, C22 += M3
            elif s == 3: # M4
                c_blocks = {0: 1, 2: 1} # C11 += M4, C21 += M4
            elif s == 4: # M5
                c_blocks = {0: -1, 1: 1} # C11 -= M5, C12 += M5
            elif s == 5: # M6
                c_blocks = {3: 1} # C22 += M6
            elif s == 6: # M7
                c_blocks = {0: 1} # C11 += M7
            
            # Now, for each affected C block, we add the contribution of P_t.
            # The entry M_s[bi, bj] contributes to C_block[bi, bj].
            # C_block is a 2x2 matrix.
            # C_block[bi, bj] corresponds to C[2*bi + i, 2*bj + j].
            
            for c_block_idx, sign in c_blocks.items():
                # c_block_idx: 0=C11, 1=C12, 2=C21, 3=C22
                cb_i = c_block_idx // 2
                cb_j = c_block_idx % 2
                
                # The value of M_s[cb_i, cb_j] is a linear combination of P_t.
                # We need the coefficient of P_t in M_s[cb_i, cb_j].
                
                # M_s[0,0] = P0 + P3 - P4 + P6
                # M_s[0,1] = P2 + P4
                # M_s[1,0] = P1 + P3
                # M_s[1,1] = P0 - P1 + P2 + P5
                
                coeff_P = 0
                if cb_i == 0 and cb_j == 0:
                    if t == 0: coeff_P = 1
                    elif t == 3: coeff_P = 1
                    elif t == 4: coeff_P = -1
                    elif t == 6: coeff_P = 1
                elif cb_i == 0 and cb_j == 1:
                    if t == 2: coeff_P = 1
                    elif t == 4: coeff_P = 1
                elif cb_i == 1 and cb_j == 0:
                    if t == 1: coeff_P = 1
                    elif t == 3: coeff_P = 1
                elif cb_i == 1 and cb_j == 1:
                    if t == 0: coeff_P = 1
                    elif t == 1: coeff_P = -1
                    elif t == 2: coeff_P = 1
                    elif t == 5: coeff_P = 1
                
                if coeff_P != 0:
                    # Update W for all entries in this C block
                    for i in range(2):
                        for j in range(2):
                            # Global C index: k, i
                            # C is 5x4. k is row, i is col.
                            # C_block is at row 2*cb_i + i, col 2*cb_j + j?
                            # No, C is 4x4 for the first part.
                            # C[i, k]. i is row, k is col.
                            # Block C11 is rows 0,1 and cols 0,1.
                            # So C[2*cb_i + i, 2*cb_j + j].
                            # W index: k*n + i = k*4 + i.
                            # Here k = 2*cb_j + j, i = 2*cb_i + i.
                            
                            k_global = 2 * cb_j + j
                            i_global = 2 * cb_i + i
                            
                            W[k_global * 4 + i_global][idx] = sign * coeff_P

    # --- Part 2: 4x4x1 Multiplication (Rank 4) ---
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Terms: t = 49 + j for j in 0..3
    for j in range(4):
        idx = 49 + j
        for i in range(4):
            U[i * 4 + j][idx] = 1
            V[j * 5 + 4][idx] = 1
            W[4 * 4 + i][idx] = 1 # k=4, i=i. W[4*4+i] = W[16+i]

    return {"U": U, "V": V, "W": W}
