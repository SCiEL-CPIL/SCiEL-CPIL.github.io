import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen factors for 2x2 matrix multiplication
    # U_in, V_in, W_in are lists of 7 vectors of length 4
    # Indices: 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
    
    U_in = [
        [1, 0, 0, 1], # P1: a+d
        [0, 0, 1, 1], # P2: c+d
        [1, 0, 0, 0], # P3: a
        [0, 0, 0, 1], # P4: d
        [1, 1, 0, 0], # P5: a+b
        [1, 0, -1, 0],# P6: c-a
        [0, 1, 0, -1] # P7: b-d
    ]
    V_in = [
        [1, 1, 0, 0], # P1: e+h
        [1, 0, 0, 0], # P2: e
        [0, 1, 0, -1],# P3: f-h
        [0, -1, 1, 0],# P4: g-e
        [0, 0, 0, 1], # P5: h
        [1, 1, 0, 0], # P6: e+f
        [0, 0, 1, 1]  # P7: g+h
    ]
    W_in = [
        [1, 0, 0, 1], # C11 += P1, C22 += P1
        [0, 0, 1, 0], # C21 += P2
        [0, 1, 0, 0], # C12 += P3
        [1, 0, 1, 0], # C11 += P4, C21 += P4
        [-1, 1, 0, 0],# C11 -= P5, C12 += P5
        [0, 0, 0, 1], # C22 += P6
        [1, 0, 0, -1] # C11 += P7, C22 -= P7
    ]
    
    # Outer Strassen factors for 2x2 block multiplication
    # U_out, V_out, W_out are lists of 7 vectors of length 4 (blocks)
    
    U_out = [
        [1, 0, 0, 1], # A11 + A22
        [0, 0, 1, 1], # A21 + A22
        [1, 0, 0, 0], # A11
        [0, 0, 0, 1], # A22
        [1, 1, 0, 0], # A11 + A12
        [1, 0, -1, 0],# A11 - A21
        [0, 1, 0, -1] # A12 - A22
    ]
    V_out = [
        [1, 1, 0, 0], # B11 + B12
        [1, 0, 0, 0], # B11
        [0, 1, 0, -1],# B12 - B22
        [0, -1, 1, 0],# B21 - B11
        [0, 0, 0, 1], # B22
        [1, 1, 0, 0], # B11 + B12
        [0, 0, 1, 1]  # B21 + B22
    ]
    W_out = [
        [1, 0, 0, 1], # C11 += P1, C22 += P1
        [0, 0, 1, 0], # C21 += P2
        [0, 1, 0, 0], # C12 += P3
        [1, 0, 1, 0], # C11 += P4, C21 += P4
        [-1, 1, 0, 0],# C11 -= P5, C12 += P5
        [0, 0, 0, 1], # C22 += P6
        [1, 0, 0, -1] # C11 += P7, C22 -= P7
    ]

    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    r = 0
    for k in range(7):
        for j in range(7):
            # Construct U[:, r]
            # U_out[k] has 4 entries for blocks 0,1,2,3
            # Block b is 2x2.
            # U_in[j] has 4 entries for elements of the 2x2 matrix.
            
            for b_idx in range(4):
                coeff_out = U_out[k][b_idx]
                if coeff_out != 0:
                    # Block b_idx corresponds to 4 elements in A
                    # Block 0: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,4,5
                    # Block 1: (0,2), (0,3), (1,2), (1,3) -> indices 2,3,6,7
                    # Block 2: (2,0), (2,1), (3,0), (3,1) -> indices 8,9,12,13
                    # Block 3: (2,2), (2,3), (3,2), (3,3) -> indices 10,11,14,15
                    base = b_idx * 4
                    for e_idx in range(4):
                        coeff_in = U_in[j][e_idx]
                        if coeff_in != 0:
                            # Map local element e_idx in block to global index
                            # Local 0: (0,0) -> global (row, col)
                            # Block 0: rows 0,1 cols 0,1
                            # Block 1: rows 0,1 cols 2,3
                            # Block 2: rows 2,3 cols 0,1
                            # Block 3: rows 2,3 cols 2,3
                            
                            # e_idx 0: (0,0)
                            # e_idx 1: (0,1)
                            # e_idx 2: (1,0)
                            # e_idx 3: (1,1)
                            
                            r_local = e_idx // 2
                            c_local = e_idx % 2
                            
                            # Global row
                            if b_idx < 2:
                                g_row = r_local
                            else:
                                g_row = 2 + r_local
                                
                            # Global col
                            if b_idx % 2 == 0:
                                g_col = c_local
                            else:
                                g_col = 2 + c_local
                                
                            g_idx = g_row * 4 + g_col
                            U[g_idx][r] += coeff_out * coeff_in
                            
            # Construct V[:, r]
            for b_idx in range(4):
                coeff_out = V_out[k][b_idx]
                if coeff_out != 0:
                    base = b_idx * 4
                    for e_idx in range(4):
                        coeff_in = V_in[j][e_idx]
                        if coeff_in != 0:
                            r_local = e_idx // 2
                            c_local = e_idx % 2
                            
                            if b_idx < 2:
                                g_row = r_local
                            else:
                                g_row = 2 + r_local
                                
                            if b_idx % 2 == 0:
                                g_col = c_local
                            else:
                                g_col = 2 + c_local
                                
                            g_idx = g_row * 4 + g_col
                            V[g_idx][r] += coeff_out * coeff_in
                            
            # Construct W[:, r]
            # W_out[k] is for the 2x2 block result M_k
            # W_in[j] is for the 2x2 element result
            # The result M_k is a 2x2 block matrix.
            # W_out[k][b_idx] is the coefficient for block b_idx of C.
            # W_in[j][e_idx] is the coefficient for element e_idx of that block.
            
            for b_idx in range(4):
                coeff_out = W_out[k][b_idx]
                if coeff_out != 0:
                    for e_idx in range(4):
                        coeff_in = W_in[j][e_idx]
                        if coeff_in != 0:
                            r_local = e_idx // 2
                            c_local = e_idx % 2
                            
                            if b_idx < 2:
                                g_row = r_local
                            else:
                                g_row = 2 + r_local
                                
                            if b_idx % 2 == 0:
                                g_col = c_local
                            else:
                                g_col = 2 + c_local
                                
                            g_idx = g_row * 4 + g_col
                            W[g_idx][r] += coeff_out * coeff_in
                            
            r += 1
            
    return {"U": U, "V": V, "W": W}
