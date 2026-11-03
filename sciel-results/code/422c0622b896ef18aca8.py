import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists initially for ease of construction,
    # then convert to numpy or return as lists if allowed.
    # Task says "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # and "integer/half-integer entries".
    
    # Rank 1: Strassen 4x4x4 part (cols 0-3)
    # Rank 2: Schoolbook 4x4x3 part (cols 4-6)
    
    # Pre-allocate for max possible rank (112) to be safe, then slice
    max_rank = n * m * p
    U = [[0.0] * max_rank for _ in range(n * m)]
    V = [[0.0] * max_rank for _ in range(m * p)]
    W = [[0.0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Blocks of A: 00, 01, 10, 11
    # A_00: rows 0,1 cols 0,1
    # A_01: rows 0,1 cols 2,3
    # A_10: rows 2,3 cols 0,1
    # A_11: rows 2,3 cols 2,3
    
    # Define the 7 Strassen block combinations for A (X) and B (Y)
    # Each is a 2x2 matrix of coefficients for the A/B blocks.
    # Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # X_m: coefficients for A blocks
    # Y_m: coefficients for B blocks
    
    # M1: X = A00 + A11, Y = B00 + B11
    # M2: X = A10 + A11, Y = B00
    # M3: X = A00,        Y = B01 - B11
    # M4: X = A11,        Y = B10 - B00
    # M5: X = A00 + A01,  Y = B11
    # M6: X = A10 - A00,  Y = B00 + B01
    # M7: X = A01 - A11,  Y = B10 + B11
    
    # We represent X_m and Y_m as 2x2 arrays of 2x2 coefficient matrices?
    # No, X_m is a 2x2 matrix where each entry is a linear combination of A's entries.
    # Actually, X_m is formed by summing specific A blocks.
    # Let's define the coefficients for the 16 entries of A for each of the 4 entries of X_m.
    
    # A entry (i,j) maps to block b_r, b_c where b_r=i//2, b_c=j//2.
    # Local index li=i%2, lj=j%2.
    
    # For a given X_m entry (li, lj), it is a sum of A blocks.
    # If X_m uses A00, then for li=0,lj=0, coeff of A[0,0] is 1.
    
    # Let's define the block structure for X and Y for each of the 7 terms.
    # term_def: (X_blocks, Y_blocks) where X_blocks is a dict {block_idx: coeff}
    # X_blocks: {0: 1, 3: 1} means X = A00 + A11.
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # For each of the 7 terms, we compute 7 scalar products (p0..p6)
    # for the 2x2 matrix multiplication X*Y.
    
    # The 7 scalar products for 2x2 mat mult (Strassen):
    # p0 = (x00+x11)(y00+y11)
    # p1 = (x10+x11)y00
    # p2 = x00(y01-y11)
    # p3 = x11(y10-y00)
    # p4 = (x00+x01)y11
    # p5 = (x10-x00)(y00+y01)
    # p6 = (x01-x11)(y10+y11)
    
    # We need to expand these into linear combinations of A and B entries.
    # For a specific scalar product definition (e.g., p0), we determine:
    # - The linear combination of A entries (coeffs for U)
    # - The linear combination of B entries (coeffs for V)
    
    # Helper to get block index from (li, lj) and block_def
    # Actually, we just iterate through the 4 positions of the 2x2 X and Y matrices.
    
    for m_idx in range(7):
        x_def, y_def = strassen_terms[m_idx]
        
        # Define the 7 sub-terms for 2x2 mult
        # Each sub-term t has a definition on x_entries and y_entries
        # x_entries: dict {(li,lj): coeff} relative to the 2x2 X matrix
        # y_entries: dict {(li,lj): coeff} relative to the 2x2 Y matrix
        
        sub_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # p0
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # p1
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # p2
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # p3
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # p4
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # p5
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # p6
        ]
        
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            
            x_sub, y_sub = sub_terms[t_idx]
            
            # Construct U vector for this term
            # U index: i*m + j (i,j in 0..3)
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p) # Note: V is for B (4x7), so index j*7+k
            
            # For each (li, lj) in x_sub, we have a coefficient c_x.
            # This (li, lj) corresponds to a position in the 2x2 block X.
            # The block X is formed by summing A blocks according to x_def.
            # So, for a specific (li, lj) in X, the value is:
            # sum_{b in x_def} x_def[b] * A_block_b[li, lj]
            # A_block_b[li, lj] corresponds to A entry (2*(b//2)+li, 2*(b%2)+lj)?
            # No. Block index b: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
            # Block 0 covers A[0:2, 0:2]. So A[li, lj].
            # Block 1 covers A[0:2, 2:4]. So A[li, lj+2].
            # Block 2 covers A[2:4, 0:2]. So A[li+2, lj].
            # Block 3 covers A[2:4, 2:4]. So A[li+2, lj+2].
            
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    # A entry index
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            
            # Similarly for V (B entries)
            # B is 4x7. We are dealing with the 4x4 part (cols 0-3).
            # Block structure of B (first 4 cols):
            # B_00: rows 0:2, cols 0:2
            # B_01: rows 0:2, cols 2:4
            # B_10: rows 2:4, cols 0:2
            # B_11: rows 2:4, cols 2:4
            
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li # Row in B
                    j = 2 * bj + lj # Col in B (0..3)
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            
            # Assign to U and V
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            # Construct W vector
            # W index: k*n + i (k in 0..3, i in 0..3)
            # The output of this term contributes to the C blocks.
            # We need to map the scalar product p_{m,t} to the final C entries.
            # First, p_{m,t} is part of the product X_m Y_m = C_m (2x2 matrix).
            # C_m entries are linear combos of p_{m,0..6}.
            # Then C (4x4) entries are linear combos of C_m blocks.
            
            w_coeffs = [0.0] * (p*n)
            
            # Step 1: Determine which C_m entries use this p_t.
            # C_m = [c00, c01; c10, c11]
            # c00 = p0 + p3 - p4 + p6
            # c01 = p2 + p4
            # c10 = p1 + p3
            # c11 = p0 - p1 + p2 + p5
            
            # Coeffs of p_t in C_m entries:
            cm_coeffs = {
                (0,0): 0.0,
                (0,1): 0.0,
                (1,0): 0.0,
                (1,1): 0.0
            }
            
            if t_idx == 0: # p0
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1: # p1
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2: # p2
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3: # p3
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4: # p4
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5: # p5
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6: # p6
                cm_coeffs[(0,0)] += 1.0
            
            # Step 2: Determine which C (4x4) entries use which C_m blocks.
            # C_00 = M1 + M4 - M5 + M7
            # C_01 = M3 + M5
            # C_10 = M2 + M4
            # C_11 = M1 - M2 + M3 + M6
            
            # m_idx corresponds to M_{m_idx+1}
            # C_block (br, bc) uses M_m with coeff:
            
            c_block_coeffs = {}
            if m_idx == 0: # M1
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1: # M2
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2: # M3
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3: # M4
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4: # M5
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5: # M6
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6: # M7
                c_block_coeffs = {(0,0): 1.0}
            
            # Step 3: Combine
            # For each C_m entry (li, lj) with coeff cm_coeff:
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                # This C_m entry contributes to C block (br, bc) ?
                # No, C_m is the result of the m-th Strassen term.
                # The entries of C_m are placed into the C blocks according to c_block_coeffs.
                # Wait. The formula is:
                # C_00 = M1 + M4 - M5 + M7.
                # This means the *entire 2x2 matrix* M1 is added to the top-left 2x2 block of C.
                # So, C[br, bc] block = sum_{m} c_block_coeff[br,bc,m] * M_m.
                
                # So, for a specific local position (li, lj) in the block:
                # C[2*br+li, 2*bc+lj] = sum_{m} c_block_coeff[br,bc,m] * (M_m)_{li,lj}
                
                # We are at specific m_idx and t_idx.
                # (M_m)_{li,lj} = cm_c * p_t (plus other p's, but we are handling p_t's contribution).
                
                # So we iterate over all blocks (br, bc) that use this M_m.
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    # Final C entry index
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj # This is the column index in C (0..3)
                    
                    # W index is k*n + i
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    # --- Part 2: 4x4x3 using Schoolbook (Rank 48) ---
    # Cols 4, 5, 6 of B.
    # C[i, k] = sum_j A[i,j] B[j,k]
    # k in 4..6
    for i in range(n):
        for j in range(m):
            for k in range(4, 7): # 4, 5, 6
                r = current_rank
                current_rank += 1
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0

    # Trim to actual rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
