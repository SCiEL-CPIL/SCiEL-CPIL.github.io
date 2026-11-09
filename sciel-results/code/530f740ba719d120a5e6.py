import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    # Rank = 8 blocks * 7 Strassen terms = 56
    R = 56
    
    # Initialize U, V, W
    # U: (n*m, R) = (8, 56)
    # V: (m*p, R) = (32, 56)
    # W: (p*n, R) = (16, 56)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # We define the linear forms for U (on X) and V (on Y) for each M
    # X indices: 00, 01, 10, 11 (row, col in 2x2)
    # Y indices: 00, 01, 10, 11
    
    # For each Strassen term t in 0..6:
    # U_t is a vector of length 4 (for the 4 elements of the 2x2 A block)
    # V_t is a vector of length 4 (for the 4 elements of the 2x2 B block)
    # W_t is a vector of length 4 (for the 4 elements of the 2x2 C block)
    
    U_strassen = [
        [1, 0, 0, 1],  # M1: X11+X22
        [0, 0, 1, 1],  # M2: X21+X22
        [1, 0, 0, 0],  # M3: X11
        [0, 0, 0, 1],  # M4: X22
        [1, 1, 0, 0],  # M5: X11+X12
        [-1, 0, 1, 0], # M6: X21-X11
        [0, 1, 0, -1]  # M7: X12-X22
    ]
    
    V_strassen = [
        [1, 0, 0, 1],  # M1: Y11+Y22
        [1, 0, 0, 0],  # M2: Y11
        [0, 1, 0, -1], # M3: Y12-Y22
        [0, -1, 1, 0], # M4: Y21-Y11
        [0, 0, 0, 1],  # M5: Y22
        [1, 1, 0, 0],  # M6: Y11+Y12
        [0, 0, 1, 1]   # M7: Y21+Y22
    ]
    
    W_strassen = [
        [1, 0, 0, 1],  # M1: Z11, Z22
        [0, 0, 1, -1], # M2: Z21, -Z22
        [0, 1, 0, 1],  # M3: Z12, Z22
        [1, 0, 1, 0],  # M4: Z11, Z21
        [-1, 1, 0, 0], # M5: -Z11, Z12
        [0, 0, 0, 1],  # M6: Z22
        [1, 0, 0, 0]   # M7: Z11
    ]
    
    # Map blocks to global indices
    # Block b in 0..7
    # b = 0,1,2,3 use A1 (cols 0,1 of A)
    # b = 4,5,6,7 use A2 (cols 2,3 of A)
    
    # A is 2x4.
    # A1 is A[:, 0:2]. Indices in A: (i, 0), (i, 1).
    # A2 is A[:, 2:4]. Indices in A: (i, 2), (i, 3).
    
    # B is 4x8.
    # For b in 0..3: B block is B[0:2, 2*(b%4) : 2*(b%4)+2].
    #   Rows 0,1. Cols 2j, 2j+1 where j = b%4.
    # For b in 4..7: B block is B[2:4, 2*((b-4)%4) : 2*((b-4)%4)+2].
    #   Rows 2,3. Cols 2j, 2j+1 where j = (b-4)%4.
    
    # C is 2x8.
    # For b in 0..3: C block is C[:, 2*(b%4) : 2*(b%4)+2].
    # For b in 4..7: C block is C[:, 2*((b-4)%4) : 2*((b-4)%4)+2].
    
    for b in range(8):
        # Determine A block source indices
        if b < 4:
            # A1
            a_col_offset = 0
            b_row_offset = 0
            c_col_offset = b * 2
        else:
            # A2
            a_col_offset = 2
            b_row_offset = 2
            c_col_offset = (b - 4) * 2
            
        # Determine B block source indices
        b_col_offset = c_col_offset # Same column offset for B and C
        
        # Indices in the 2x2 local block:
        # local (0,0) -> global A(i, a_col_offset)
        # local (0,1) -> global A(i, a_col_offset + 1)
        # local (1,0) -> global A(i+1, a_col_offset)
        # local (1,1) -> global A(i+1, a_col_offset + 1)
        
        # Local B:
        # local (0,0) -> global B(b_row_offset, b_col_offset)
        # local (0,1) -> global B(b_row_offset, b_col_offset + 1)
        # local (1,0) -> global B(b_row_offset + 1, b_col_offset)
        # local (1,1) -> global B(b_row_offset + 1, b_col_offset + 1)
        
        # Local C:
        # local (0,0) -> global C(0, c_col_offset)
        # local (0,1) -> global C(0, c_col_offset + 1)
        # local (1,0) -> global C(1, c_col_offset)
        # local (1,1) -> global C(1, c_col_offset + 1)
        
        for t in range(7):
            r = b * 7 + t
            
            # Fill U
            # U_strassen[t] has coeffs for local X indices 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            # Global A index for local (i_loc, j_loc) is i_loc * m + (a_col_offset + j_loc)
            # Note: m=4.
            for i_loc in range(2):
                for j_loc in range(2):
                    local_idx = i_loc * 2 + j_loc
                    coeff = U_strassen[t][local_idx]
                    if coeff != 0:
                        global_i = i_loc
                        global_k = a_col_offset + j_loc
                        U[global_i * m + global_k, r] = coeff
                        
            # Fill V
            # V_strassen[t] has coeffs for local Y indices 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            # Global B index for local (i_loc, j_loc) is (b_row_offset + i_loc) * p + (b_col_offset + j_loc)
            # Note: p=8.
            for i_loc in range(2):
                for j_loc in range(2):
                    local_idx = i_loc * 2 + j_loc
                    coeff = V_strassen[t][local_idx]
                    if coeff != 0:
                        global_k = b_row_offset + i_loc
                        global_l = b_col_offset + j_loc
                        V[global_k * p + global_l, r] = coeff
                        
            # Fill W
            # W_strassen[t] has coeffs for local Z indices 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            # Global C index for local (i_loc, j_loc) is (0, c_col_offset + j_loc) and (1, c_col_offset + j_loc)
            # W index convention: W[k*n + i, r] where k is col of C (l) and i is row of C.
            # So index is (c_col_offset + j_loc) * n + i_loc
            for i_loc in range(2):
                for j_loc in range(2):
                    local_idx = i_loc * 2 + j_loc
                    coeff = W_strassen[t][local_idx]
                    if coeff != 0:
                        global_l = c_col_offset + j_loc
                        global_i = i_loc
                        W[global_l * n + global_i, r] = coeff
                        
    return {"U": U, "V": V, "W": W}
