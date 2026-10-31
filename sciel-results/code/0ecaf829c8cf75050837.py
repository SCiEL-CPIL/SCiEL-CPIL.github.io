import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    
    Rank = 49.
    """
    n, m, p = DIMENSIONS  # all 4
    
    # Strassen factors for 2x2 matrix multiplication
    # U: (2x2, 7), V: (2x2, 7), W: (2x2, 7)
    # Indices: (row, col)
    
    # We define the 7 terms of Strassen for 2x2 matrices.
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # U_factors[term][row][col]
    U_2x2 = np.zeros((4, 7), dtype=int)
    V_2x2 = np.zeros((4, 7), dtype=int)
    W_2x2 = np.zeros((4, 7), dtype=int)
    
    # Term 1: (0,0)+(1,1)
    U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
    V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
    W_2x2[0, 0] = 1; W_2x2[3, 0] = 1
    
    # Term 2: (1,0)+(1,1) * (0,0)
    U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
    V_2x2[0, 1] = 1
    W_2x2[2, 1] = 1; W_2x2[3, 1] = -1
    
    # Term 3: (0,0) * (0,1)-(1,1)
    U_2x2[0, 2] = 1
    V_2x2[1, 2] = 1; V_2x2[3, 2] = -1
    W_2x2[1, 2] = 1; W_2x2[3, 2] = 1
    
    # Term 4: (1,1) * (1,0)-(0,0)
    U_2x2[3, 3] = 1
    V_2x2[2, 3] = 1; V_2x2[0, 3] = -1
    W_2x2[0, 3] = 1; W_2x2[2, 3] = 1
    
    # Term 5: (0,0)+(0,1) * (1,1)
    U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
    V_2x2[3, 4] = 1
    W_2x2[0, 4] = -1; W_2x2[1, 4] = 1
    
    # Term 6: (1,0)-(0,0) * (0,0)+(0,1)
    U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
    V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
    W_2x2[3, 5] = 1
    
    # Term 7: (0,1)-(1,1) * (1,0)+(1,1)
    U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
    V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
    W_2x2[0, 6] = 1

    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Outer Strassen on 2x2 blocks of 2x2 matrices
    # The structure is identical to the inner one.
    # We can reuse the same factor matrices for the outer level.
    
    col = 0
    for outer_term in range(7):
        for inner_term in range(7):
            # Combine outer and inner factors
            # U_global[row, col] = sum( U_outer[br, bc, outer_term] * U_inner[lr, lc, inner_term] )
            # where global_row = br*2 + lr, global_col = bc*2 + lc
            
            # We can do this via broadcasting or loops.
            # U_2x2 is 4x7. Indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
            # Let's map linear index l to (r, c): r = l//2, c = l%2.
            
            # U_outer is U_2x2[:, outer_term]
            # U_inner is U_2x2[:, inner_term]
            
            # For each non-zero in U_outer and U_inner?
            # Or just multiply the matrices?
            # U_global is 16x1.
            # U_global[gr*4+gc] += U_outer[br*2+bc] * U_inner[lr*2+lc]
            
            # Let's just iterate over the 4 elements of the 2x2 factor
            for outer_idx in range(4):
                u_val = U_2x2[outer_idx, outer_term]
                if u_val == 0: continue
                ob_r = outer_idx // 2
                ob_c = outer_idx % 2
                
                for inner_idx in range(4):
                    u_val_in = U_2x2[inner_idx, inner_term]
                    if u_val_in == 0: continue
                    in_r = inner_idx // 2
                    in_c = inner_idx % 2
                    
                    gr = ob_r * 2 + in_r
                    gc = ob_c * 2 + in_c
                    U[gr * 4 + gc, col] += u_val * u_val_in
                    
                    v_val = V_2x2[outer_idx, outer_term]
                    if v_val != 0:
                        # Note: V uses the same outer/inner indices structure
                        # V_global[gr*4+gc] += V_outer[ob_r, ob_c] * V_inner[in_r, in_c]
                        pass # We need to do this separately for V and W
                    
            # Redoing V and W properly
            for outer_idx in range(4):
                v_val_out = V_2x2[outer_idx, outer_term]
                if v_val_out == 0: continue
                ob_r = outer_idx // 2
                ob_c = outer_idx % 2
                
                for inner_idx in range(4):
                    v_val_in = V_2x2[inner_idx, inner_term]
                    if v_val_in == 0: continue
                    in_r = inner_idx // 2
                    in_c = inner_idx % 2
                    
                    gr = ob_r * 2 + in_r
                    gc = ob_c * 2 + in_c
                    V[gr * 4 + gc, col] += v_val_out * v_val_in
                    
            for outer_idx in range(4):
                w_val_out = W_2x2[outer_idx, outer_term]
                if w_val_out == 0: continue
                ob_r = outer_idx // 2
                ob_c = outer_idx % 2
                
                for inner_idx in range(4):
                    w_val_in = W_2x2[inner_idx, inner_term]
                    if w_val_in == 0: continue
                    in_r = inner_idx // 2
                    in_c = inner_idx % 2
                    
                    gr = ob_r * 2 + in_r
                    gc = ob_c * 2 + in_c
                    # W index: gc * n + gr (k*n+i)
                    W[gc * 4 + gr, col] += w_val_out * w_val_in
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
