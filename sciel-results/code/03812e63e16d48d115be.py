import numpy as np

def get_m2_decomposition():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # ... (as above)
    return U, V, W

def construct():
    U2, V2, W2 = get_m2_decomposition()
    
    # Build M4 decomposition
    # U4, V4, W4 of size 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen steps
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Indices in A (16):
    # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
    # Indices in B (16):
    # B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
    # Indices in C (16):
    # C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
    
    # For each Strassen term s (0..6) and each M2 term k (0..6):
    for s in range(7):
        for k in range(7):
            col = s * 7 + k
            
            # U part: depends on A blocks
            # V part: depends on B blocks
            # W part: depends on C blocks
            
            # We need to know which blocks are involved in Strassen term s.
            # And which indices within those blocks are involved in M2 term k.
            
            # This requires knowing the block structure of U2, V2, W2.
            # U2 has 4 rows.
            # Row 0: A11 (a)
            # Row 1: A12 (b)
            # Row 2: A21 (c)
            # Row 3: A22 (d)
            
            # For Strassen term s, we form a linear combination of A blocks.
            # e.g., M1: A11 + A22.
            # So U for this term is U2[:, k] for A11 and U2[:, k] for A22.
            
            # Let's define the block coefficients for each Strassen term.
            # s=0: M1 = (A11+A22)(B11+B22)
            # s=1: M2 = (A21+A22)B11
            # s=2: M3 = A11(B12-B22)
            # s=3: M4 = A22(B21-B11)
            # s=4: M5 = (A11+A12)B22
            # s=5: M6 = (A21-A11)(B11+B12)
            # s=6: M7 = (A12-A22)(B21+B22)
            
            # Coefficients for A blocks:
            a_coeff = [
                [1, 0, 1, 0], # s=0: A11, A22
                [0, 0, 1, 1], # s=1: A21, A22
                [1, 0, 0, 0], # s=2: A11
                [0, 0, 0, 1], # s=3: A22
                [1, 1, 0, 0], # s=4: A11, A12
                [-1, 0, 1, 0], # s=5: A21, A11
                [0, 1, 0, -1]  # s=6: A12, A22
            ]
            
            # Coefficients for B blocks:
            b_coeff = [
                [1, 0, 0, 1], # s=0: B11, B22
                [1, 0, 0, 0], # s=1: B11
                [0, 1, 0, -1], # s=2: B12, B22
                [0, 0, 1, -1], # s=3: B21, B11
                [0, 0, 0, 1], # s=4: B22
                [1, 1, 0, 0], # s=5: B11, B12
                [0, 0, 1, 1]  # s=6: B21, B22
            ]
            
            # Coefficients for C blocks:
            c_coeff = [
                [1, 0, 0, 1], # s=0: C11, C22
                [0, 0, 1, -1], # s=1: C21, C22
                [0, 1, 0, 1], # s=2: C12, C22
                [1, 0, 1, 0], # s=3: C11, C21
                [-1, 1, 0, 0], # s=4: C11, C12
                [0, 0, 0, 1], # s=5: C22
                [1, 0, 0, 0]  # s=6: C11
            ]
            
            # For each block involved, add the corresponding U2, V2, W2 rows.
            # U2 rows: 0(A11), 1(A12), 2(A21), 3(A22)
            for block in range(4):
                coeff_a = a_coeff[s][block]
                coeff_b = b_coeff[s][block]
                coeff_c = c_coeff[s][block]
                
                if coeff_a != 0:
                    # Add U2[:, k] * coeff_a to U4[block*4:(block+1)*4, col]
                    U4[block*4:(block+1)*4, col] += coeff_a * U2[:, k]
                
                if coeff_b != 0:
                    V4[block*4:(block+1)*4, col] += coeff_b * V2[:, k]
                
                if coeff_c != 0:
                    W4[block*4:(block+1)*4, col] += coeff_c * W2[:, k]
            
    # Now U4, V4, W4 are 16x49
    # Construct V_full and W_full for 4x4x8
    V_full = np.zeros((32, 49))
    V_full[0:16, :] = V4
    V_full[16:32, :] = V4
    
    W_full = np.zeros((32, 49))
    W_full[0:16, :] = W4
    W_full[16:32, :] = W4
    
    return {"U": U4.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
