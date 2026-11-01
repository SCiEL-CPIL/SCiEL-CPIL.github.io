import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    
    Uses Strassen's algorithm for each of the 7 independent 2x2 blocks.
    Total rank = 49.
    
    The 2x2 matrix A multiplies a 2x14 matrix B to produce a 14x2 matrix C.
    We view B as 7 independent 2x2 blocks (columns 2r, 2r+1 of B).
    For each block, we use Strassen's 7-multiplication algorithm.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A indices: 00, 01, 10, 11 -> flat indices 0, 1, 2, 3
    U_forms = [
        [1, 0, 0, 1],   # M1 = A00 + A11
        [0, 0, 1, 1],   # M2 = A10 + A11
        [1, 0, 0, 0],   # M3 = A00
        [0, 0, 0, 1],   # M4 = A11
        [1, 1, 0, 0],   # M5 = A00 + A01
        [-1, 0, 1, 0],  # M6 = A10 - A00
        [0, 1, 0, -1]   # M7 = A01 - A11
    ]
    
    # Standard Strassen B-forms: for a 2x2 B = [[e,f],[g,h]]
    # M1*B: [[e, h], [g, h]] -> V coeffs for (e,f,g,h): (1,0,0,1)
    # M2*B: [[e, 0], [g, 0]] -> (1,0,1,0)
    # M3*B: [[f, h], [0, h]] -> (0,1,0,1)
    # M4*B: [[e, 0], [g, 0]] -> wait, M4 = A11, B is same -> (1,0,1,0)? No.
    # Let me re-derive Strassen properly.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A-forms (rows of U for the 4 A-entries):
    # M1: [1,0,0,1]
    # M2: [0,0,1,1]
    # M3: [1,0,0,0]
    # M4: [0,0,0,1]
    # M5: [1,1,0,0]
    # M6: [-1,0,1,0]
    # M7: [0,1,0,-1]
    #
    # B-forms (coeffs for B00=e, B01=f, B10=g, B11=h):
    # M1: B00+B11 -> [1,0,0,1]
    # M2: B00 -> [1,0,0,0]
    # M3: B01-B11 -> [0,1,0,-1]
    # M4: B10-B00 -> [-1,0,1,0]
    # M5: B11 -> [0,0,0,1]
    # M6: B00+B01 -> [1,1,0,0]
    # M7: B10+B11 -> [0,0,1,1]
    #
    # C-reconstruction (coeffs for C00, C01, C10, C11):
    # M1: C00+=1, C11+=1 -> [1,0,0,1]
    # M2: C10+=1, C11+=-1 -> [0,0,1,-1]
    # M3: C01+=1, C11+=1 -> [0,1,0,1]
    # M4: C00+=1, C10+=1 -> [1,0,1,0]
    # M5: C00+=-1, C01+=1 -> [-1,1,0,0]
    # M6: C11+=1 -> [0,0,0,1]
    # M7: C00+=1 -> [1,0,0,0]
    
    B_forms = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    C_forms = [
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, -1],  # M2: C10+1, C11-1
        [0, 1, 0, 1],   # M3: C01+1, C11+1
        [1, 0, 1, 0],   # M4: C00+1, C10+1
        [-1, 1, 0, 0],  # M5: C00-1, C01+1
        [0, 0, 0, 1],   # M6: C11+1
        [1, 0, 0, 0]    # M7: C00+1
    ]
    
    for r in range(num_blocks):
        # B block r: columns 2r, 2r+1
        # B is 2x14. Flat index for B[j][k] = j*p + k
        idx_e = 0 * p + 2 * r       # B[0, 2r]
        idx_f = 0 * p + 2 * r + 1   # B[0, 2r+1]
        idx_g = 1 * p + 2 * r       # B[1, 2r]
        idx_h = 1 * p + 2 * r + 1   # B[1, 2r+1]
        
        # C block r: rows 2r, 2r+1
        # C is 14x2. Flat index for C[k][i] = k*n + i
        w_c00 = (2 * r) * n + 0     # C[2r, 0]
        w_c01 = (2 * r) * n + 1     # C[2r, 1]
        w_c10 = (2 * r + 1) * n + 0 # C[2r+1, 0]
        w_c11 = (2 * r + 1) * n + 1 # C[2r+1, 1]
        
        b_cols = [idx_e, idx_f, idx_g, idx_h]  # e, f, g, h
        c_rows = [w_c00, w_c01, w_c10, w_c11]  # C00, C01, C10, C11
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U: A-forms
            for i in range(4):
                U[i][c] = float(U_forms[s][i])
            
            # Fill V: B-forms
            for i in range(4):
                coeff = float(B_forms[s][i])
                if coeff != 0.0:
                    V[b_cols[i]][c] = coeff
            
            # Fill W: C-reconstruction
            for i in range(4):
                coeff = float(C_forms[s][i])
                if coeff != 0.0:
                    W[c_rows[i]][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
