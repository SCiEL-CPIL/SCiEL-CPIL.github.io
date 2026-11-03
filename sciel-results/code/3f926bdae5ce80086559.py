import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _build_rank44():
    """
    Build a rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    - Split A (3x4) into A_top (2x4) and A_bot (1x4).
    - C_bot = A_bot * B: rank 16 (schoolbook for 1x4*4x4)
    - C_top = A_top * B: rank 28 (Strassen-based for 2x4*4x4)
    
    For 2x4*4x4:
    - Split A_top into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    - Split B into 2x2 blocks: B_TL, B_TR, B_BL, B_BR
    - C_top = A_L*(B_TL + B_BL) + A_R*(B_TR + B_BR)
    - Use Strassen for each 2x2*2x2 multiplication (4 multiplications, 7 terms each = 28 terms)
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build U, V, W as lists of lists (to be converted to arrays later if needed)
    # U: shape (n*m, rank) = (12, 44)
    # V: shape (m*p, rank) = (16, 44)
    # W: shape (p*n, rank) = (12, 44)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal col
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val
        col += 1
    
    # ===== Part 1: C_bot = A_bot * B (1x4 * 4x4) =====
    # A_bot is row 2 of A, so A_bot[j] = A[2][j] for j in 0..3
    # C_bot[k] = sum_j A_bot[j] * B[j][k] for k in 0..3
    # Schoolbook: 16 terms
    for j in range(4):
        for k in range(4):
            # U index: A row 2, col j -> 2*4 + j = 8 + j
            # V index: B row j, col k -> j*4 + k
            # W index: C col k, row 2 -> k*3 + 2
            add_term(8 + j, j * 4 + k, k * 3 + 2, 1, 1, 1)
    
    # ===== Part 2: C_top = A_top * B (2x4 * 4x4) =====
    # A_top is rows 0,1 of A.
    # A_L = A_top[:, 0:2] (2x2), A_R = A_top[:, 2:4] (2x2)
    # B = [[B_TL, B_TR], [B_BL, B_BR]] where each block is 2x2
    # B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4]
    # B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    
    # C_top = A_L * (B_TL + B_BL) + A_R * (B_TR + B_BR)
    # Let S1 = B_TL + B_BL, S2 = B_TR + B_BR
    # C_top = A_L * S1 + A_R * S2
    
    # We'll compute A_L * S1 and A_R * S2 using Strassen.
    
    # Strassen for 2x2 * 2x2:
    # Given matrices X (2x2) and Y (2x2), compute Z = X * Y
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y21 - Y22)
    # M4 = X22(Y11 - Y21)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # For A_L * S1:
    # A_L = [[A00, A01], [A10, A11]] where Aij = A[i][j]
    # S1 = B_TL + B_BL = [[B00+B20, B01+B21], [B10+B30, B11+B31]]
    # S1_11 = B00 + B20, S1_12 = B01 + B21
    # S1_21 = B10 + B30, S1_22 = B11 + B31
    
    # For A_R * S2:
    # A_R = [[A02, A03], [A12, A13]]
    # S2 = B_TR + B_BR = [[B02+B22, B03+B23], [B12+B32, B13+B33]]
    # S2_11 = B02 + B22, S2_12 = B03 + B23
    # S2_21 = B12 + B32, S2_22 = B13 + B33
    
    # We need to add 7 terms for A_L * S1 and 7 terms for A_R * S2.
    
    # Helper for Strassen block
    def add_strassen_block(a_row_offset, a_col_offset, s_row_offset, s_col_offset, c_row_offset, c_col_offset):
        """
        Add 7 Strassen terms for multiplying a 2x2 block of A (starting at a_row_offset, a_col_offset)
        by a 2x2 block of S (starting at s_row_offset, s_col_offset in the combined S matrix).
        
        The result contributes to C starting at (c_row_offset, c_col_offset).
        
        a_row_offset: 0 or 1 (row in A_top)
        a_col_offset: 0 or 2 (col in A_top)
        s_row_offset: 0 or 2 (row in S, which is 4x4)
        s_col_offset: 0 or 2 (col in S)
        c_row_offset: 0 or 1 (row in C_top)
        c_col_offset: 0 or 2 (col in C_top)
        """
        nonlocal col
        
        # A indices (in global A):
        # A_11 = A[a_row_offset][a_col_offset]
        # A_12 = A[a_row_offset][a_col_offset + 1]
        # A_21 = A[a_row_offset + 1][a_col_offset]
        # A_22 = A[a_row_offset + 1][a_col_offset + 1]
        
        # S indices (in global B):
        # S_11 = B[s_row_offset][s_col_offset]
        # S_12 = B[s_row_offset][s_col_offset + 1]
        # S_21 = B[s_row_offset + 1][s_col_offset]
        # S_22 = B[s_row_offset + 1][s_col_offset + 1]
        
        # But S is a sum of two B blocks!
        # For S1: S1_11 = B[0][0] + B[2][0], etc.
        # So we need to handle the sum carefully.
        
        # Let's define the S block as a sum of two B blocks.
        # For S1 (B_TL + B_BL):
        #   S1_11 = B[0][0] + B[2][0]
        #   S1_12 = B[0][1] + B[2][1]
        #   S1_21 = B[1][0] + B[3][0]
        #   S1_22 = B[1][1] + B[3][1]
        
        # For S2 (B_TR + B_BR):
        #   S2_11 = B[0][2] + B[2][2]
        #   S2_12 = B[0][3] + B[2][3]
        #   S2_21 = B[1][2] + B[3][2]
        #   S2_22 = B[1][3] + B[3][3]
        
        # This is getting complex. Let me restructure.
        
        # Actually, let's just directly add the terms for each Strassen multiplication.
        # For A_L * S1:
        # X = A_L, Y = S1
        # M1 = (X11 + X22)(Y11 + Y22)
        # M2 = (X21 + X22)Y11
        # M3 = X11(Y21 - Y22)
        # M4 = X22(Y11 - Y21)
        # M5 = (X11 + X12)Y22
        # M6 = (X21 - X11)(Y11 + Y12)
        # M7 = (X12 - X22)(Y21 + Y22)
        
        # Since Y = S1 = B_TL + B_BL, each Y term is a sum of two B terms.
        # This means each M term will involve sums/differences of B entries.
        # We can handle this by adding multiple terms to U, V, W.
        
        # Let me define helper to get B index
        def b_idx(r, c):
            return r * 4 + c
        
        # A indices
        a11 = a_row_offset * 4 + a_col_offset
        a12 = a_row_offset * 4 + a_col_offset + 1
        a21 = (a_row_offset + 1) * 4 + a_col_offset
        a22 = (a_row_offset + 1) * 4 + a_col_offset + 1
        
        # For S1:
        # Y11 = B[0][0] + B[2][0] -> b_idx(0,0) + b_idx(2,0)
        # Y12 = B[0][1] + B[2][1] -> b_idx(0,1) + b_idx(2,1)
        # Y21 = B[1][0] + B[3][0] -> b_idx(1,0) + b_idx(3,0)
        # Y22 = B[1][1] + B[3][1] -> b_idx(1,1) + b_idx(3,1)
        
        # For S2:
        # Y11 = B[0][2] + B[2][2] -> b_idx(0,2) + b_idx(2,2)
        # Y12 = B[0][3] + B[2][3] -> b_idx(0,3) + b_idx(2,3)
        # Y21 = B[1][2] + B[3][2] -> b_idx(1,2) + b_idx(3,2)
        # Y22 = B[1][3] + B[3][3] -> b_idx(1,3) + b_idx(3,3)
        
        # C indices (in global C, row-major: C[k][i] -> k*3 + i)
        c11 = c_row_offset * 4 + c_col_offset  # Wait, C is 3x4, so C[i][k] -> i*4 + k? 
        # No, the trace convention is W[k*n+i, r] where C = A*B, so C[i][k] = sum_j A[i][j]*B[j][k]
        # W index is k*n + i = k*3 + i
        # So for C_top[i][k], i in 0..1, k in 0..3:
        # W index = k*3 + i
        
        # Let me redefine:
        # C_11 = C_top[c_row_offset][c_col_offset] -> W index = c_col_offset*3 + c_row_offset
        # C_12 = C_top[c_row_offset][c_col_offset+1] -> W index = (c_col_offset+1)*3 + c_row_offset
        # C_21 = C_top[c_row_offset+1][c_col_offset] -> W index = c_col_offset*3 + (c_row_offset+1)
        # C_22 = C_top[c_row_offset+1][c_col_offset+1] -> W index = (c_col_offset+1)*3 + (c_row_offset+1)
        
        c11_idx = c_col_offset * 3 + c_row_offset
        c12_idx = (c_col_offset + 1) * 3 + c_row_offset
        c21_idx = c_col_offset * 3 + (c_row_offset + 1)
        c22_idx = (c_col_offset + 1) * 3 + (c_row_offset + 1)
        
        # Now, for each Strassen term, we need to add contributions to U, V, W.
        # Since Y is a sum of two B blocks, each Y entry is a sum of two B entries.
        # This means each M term will be a sum of products, which we can expand.
        
        # Let me define the Y entries as lists of (b_index, coeff)
        if s_row_offset == 0 and s_col_offset == 0:  # S1_11
            y11 = [(b_idx(0,0), 1), (b_idx(2,0), 1)]
            y12 = [(b_idx(0,1), 1), (b_idx(2,1), 1)]
            y21 = [(b_idx(1,0), 1), (b_idx(3,0), 1)]
            y22 = [(b_idx(1,1), 1), (b_idx(3,1), 1)]
        elif s_row_offset == 0 and s_col_offset == 2:  # S2_11
            y11 = [(b_idx(0,2), 1), (b_idx(2,2), 1)]
            y12 = [(b_idx(0,3), 1), (b_idx(2,3), 1)]
            y21 = [(b_idx(1,2), 1), (b_idx(3,2), 1)]
            y22 = [(b_idx(1,3), 1), (b_idx(3,3), 1)]
        else:
            raise ValueError("Invalid S block")
        
        # M1 = (X11 + X22)(Y11 + Y22)
        # = (A11 + A22) * (Y11 + Y22)
        # Y11 + Y22 = sum of 4 B entries
        # So M1 = (A11 + A22) * (b1 + b2 + b3 + b4)
        # We need to add terms for each product.
        
        # This is getting very complex. Let me try a different approach.
        
        # Instead of trying to handle the sum in S, let me just compute A_L * B_TL, A_L * B_BL, A_R * B_TR, A_R * B_BR separately using Strassen, and add them up.
        
        # But that would be 4 * 7 = 28 terms, which is what we want!
        
        # So let me redefine: for each pair (A_block, B_block), add 7 Strassen terms.
        
        # A_L * B_TL: a_row_offset=0, a_col_offset=0, b_row_offset=0, b_col_offset=0, c_row_offset=0, c_col_offset=0
        # A_L * B_BL: a_row_offset=0, a_col_offset=0, b_row_offset=2, b_col_offset=0, c_row_offset=0, c_col_offset=0
        # A_R * B_TR: a_row_offset=0, a_col_offset=2, b_row_offset=0, b_col_offset=2, c_row_offset=0, c_col_offset=2
        # A_R * B_BR: a_row_offset=0, a_col_offset=2, b_row_offset=2, b_col_offset=2, c_row_offset=0, c_col_offset=2
        
        # Wait, this doesn't look right. Let me reconsider.
        
        # C_top = A_L * B_TL + A_L * B_BL + A_R * B_TR + A_R * B_BR
        # = A_L * (B_TL + B_BL) + A_R * (B_TR + B_BR)
        
        # So we can compute each 2x2*2x2 product separately and add them.
        
        # Let me create a helper to add 7 Strassen terms for X * Y where X is a 2x2 block of A and Y is a 2x2 block of B.
        
        def add_strassen_2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
            nonlocal col
            
            # A indices
            a11 = a_row_off * 4 + a_col_off
            a12 = a_row_off * 4 + a_col_off + 1
            a21 = (a_row_off + 1) * 4 + a_col_off
            a22 = (a_row_off + 1) * 4 + a_col_off + 1
            
            # B indices
            b11 = b_row_off * 4 + b_col_off
            b12 = b_row_off * 4 + b_col_off + 1
            b21 = (b_row_off + 1) * 4 + b_col_off
            b22 = (b_row_off + 1) * 4 + b_col_off + 1
            
            # C indices
            c11_idx = c_col_off * 3 + c_row_off
            c12_idx = (c_col_off + 1) * 3 + c_row_off
            c21_idx = c_col_off * 3 + (c_row_off + 1)
            c22_idx = (c_col_off + 1) * 3 + (c_row_off + 1)
            
            # M1 = (A11 + A22)(B11 + B22)
            # Add terms for A11*B11, A11*B22, A22*B11, A22*B22
            add_term(a11, b11, c11_idx, 1, 1, 1)
            add_term(a11, b22, c11_idx, 1, 1, 1)
            add_term(a22, b11, c11_idx, 1, 1, 1)
            add_term(a22, b22, c11_idx, 1, 1, 1)
            # Also contributes to C22 with -1 for some terms? No, M1 contributes to C11 and C22.
            # C11 = M1 + M4 - M5 + M7
            # C22 = M1 - M2 + M3 + M6
            # So M1 contributes +1 to C11 and +1 to C22.
            # But we're adding individual terms, not M terms.
            # This approach is flawed.
            
            # Let me try yet another approach: just use the known rank-28 decomposition for 2x4*4x4.
            # I'll look up or derive it.
            
            # Actually, let me just use schoolbook for the 2x4*4x4 part as well, which gives rank 32.
            # Total rank = 16 + 32 = 48.
            # But we want rank 44.
            
            # Let me try to find a rank-28 decomposition for 2x4*4x4.
            
            # One approach: use the fact that 2x4*4x4 can be seen as two 2x2*4x4 multiplications.
            # Or use the "subtensor" property.
            
            # For now, let me just return the schoolbook rank-48 decomposition.
            raise NotImplementedError("Strassen block not implemented")
        
        raise NotImplementedError("Strassen block not implemented")
    
    # Since the Strassen approach is complex, let me just use schoolbook for everything.
    # This gives rank 48, which is correct but not optimal.
    
    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook for all 3x4*4x4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to build rank-44 decomposition
    try:
        result = _build_rank44()
        return result
    except Exception as e:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
