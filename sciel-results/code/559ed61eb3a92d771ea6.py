import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # 7 products
    # U, V, W for 2x2 (size 4x7)
    U_strassen = np.zeros((4, 7), dtype=int)
    V_strassen = np.zeros((4, 7), dtype=int)
    W_strassen = np.zeros((4, 7), dtype=int)
    
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W index: col*2 + row
    
    # M1 = (A00+A11)(B00+B11)
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[1, 1] = 1 # C10 += M2
    
    # M3 = A00(B11-B00)
    U_strassen[0, 2] = 1
    V_strassen[3, 2] = 1; V_strassen[0, 2] = -1
    W_strassen[2, 2] = 1 # C01 += M3
    
    # M4 = A11(B00-B01)
    U_strassen[3, 3] = 1
    V_strassen[0, 3] = 1; V_strassen[1, 3] = -1
    W_strassen[3, 3] = 1 # C11 += M4
    
    # M5 = (A00+A01)B11
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[2, 4] = 1 # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[1, 5] = 1 # C10 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[3, 6] = 1 # C11 += M7? 
    # Wait, standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M7
    
    # Let's re-verify W_strassen assignments.
    # W index: col*2+row.
    # C00 (idx 0): M1 - M4 + M5 - M7
    W_strassen[0, 0] = 1
    W_strassen[0, 3] = -1
    W_strassen[0, 4] = 1
    W_strassen[0, 6] = -1
    
    # C01 (idx 2): M3 + M5
    W_strassen[2, 2] = 1
    W_strassen[2, 4] = 1
    
    # C10 (idx 1): M2 + M6
    W_strassen[1, 1] = 1
    W_strassen[1, 5] = 1
    
    # C11 (idx 3): M1 - M2 + M4 + M7
    W_strassen[3, 0] = 1
    W_strassen[3, 1] = -1
    W_strassen[3, 3] = 1
    W_strassen[3, 6] = 1
    
    # Schoolbook 3x3 decomposition
    # 27 products
    U_school = np.zeros((9, 27), dtype=int)
    V_school = np.zeros((9, 27), dtype=int)
    W_school = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_school[i * 3 + j, col] = 1
                V_school[j * 3 + k, col] = 1
                W_school[k * 3 + i, col] = 1
                col += 1
    
    # Global matrices
    rank = 7 * 27
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # Map Strassen block indices to global indices
    # Block (bi, bj) in 2x2 grid.
    # Global row/col offset is 3*bi, 3*bj.
    
    for k_str in range(7):
        for k_school in range(27):
            c = k_str * 27 + k_school
            
            # Determine coefficients for U
            # U_strassen[bi*2+bj, k_str] gives the coefficient for block (bi, bj) in the left operand
            for bi in range(2):
                for bj in range(2):
                    coeff = U_strassen[bi * 2 + bj, k_str]
                    if coeff != 0:
                        # Block (bi, bj) occupies rows 3*bi..3*bi+2, cols 3*bj..3*bj+2
                        for ui in range(3):
                            for uj in range(3):
                                global_idx = (3 * bi + ui) * 6 + (3 * bj + uj)
                                school_idx = ui * 3 + uj
                                U[global_idx, c] += coeff * U_school[school_idx, k_school]
            
            # Determine coefficients for V
            for bi in range(2):
                for bj in range(2):
                    coeff = V_strassen[bi * 2 + bj, k_str]
                    if coeff != 0:
                        for vi in range(3):
                            for vj in range(3):
                                global_idx = (3 * bi + vi) * 6 + (3 * bj + vj)
                                school_idx = vi * 3 + vj
                                V[global_idx, c] += coeff * V_school[school_idx, k_school]
            
            # Determine coefficients for W
            # W_strassen[ci*2+cj, k_str] gives coefficient for block (ci, cj) in the output
            for ci in range(2):
                for cj in range(2):
                    coeff = W_strassen[ci * 2 + cj, k_str]
                    if coeff != 0:
                        for wi in range(3):
                            for wj in range(3):
                                # Output block (ci, cj) at local (wi, wj)
                                # Global row = 3*ci + wi, Global col = 3*cj + wj
                                # W index convention: col * n + row
                                global_row = 3 * ci + wi
                                global_col = 3 * cj + wj
                                global_idx = global_col * 6 + global_row
                                school_idx = wj * 3 + wi
                                W[global_idx, c] += coeff * W_school[school_idx, k_school]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
