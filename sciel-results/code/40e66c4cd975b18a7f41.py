import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A01-A10)(B10+B11)
    # M7 = (A00-A01)(B00+B01)
    
    # Coefficients for U (A side)
    # r=0: M1 -> A00, A11
    # r=1: M2 -> A10, A11
    # r=2: M3 -> A00
    # r=3: M4 -> A11
    # r=4: M5 -> A00, A01
    # r=5: M6 -> A01, A10
    # r=6: M7 -> A00, A01
    
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 1], # A00
        [0, 1, 0, 1, 0, 1, 0], # A01
        [0, 1, 0, 0, 0, -1, 0],# A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    
    # Coefficients for V (B side)
    # r=0: M1 -> B00, B11
    # r=1: M2 -> B00
    # r=2: M3 -> B01, B11
    # r=3: M4 -> B10, B00
    # r=4: M5 -> B11
    # r=5: M6 -> B10, B11
    # r=6: M7 -> B00, B01
    
    V_strassen = [
        [1, 1, 0, -1, 0, 0, 1], # B00
        [0, 0, 1, 0, 0, 1, 1],  # B01
        [1, 0, -1, 1, 0, 0, 0], # B10
        [1, 0, 0, 0, 1, 1, 0]   # B11
    ]
    
    # Coefficients for W (C side)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Block 1: A1 (2x2), B11 (2x2, cols 0-1), C11 (2x2, cols 0-1)
    # Global indices:
    # A: i in 0-1, j in 0-1 -> u = i*3 + j
    # B: j in 0-1, k in 0-1 -> v = j*4 + k
    # C: k in 0-1, i in 0-1 -> w = k*2 + i
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u = i * 3 + j
                U[u, r] += U_strassen[i][j]
        for j in range(2):
            for k in range(2):
                v = j * 4 + k
                V[v, r] += V_strassen[j][k]
        for k in range(2):
            for i in range(2):
                w = k * 2 + i
                W[w, r] += W_strassen[k][i] # Note: W_strassen is indexed [k][i] in my list above?
                # My W_strassen list:
                # [0] is C00 (i=0, k=0)
                # [1] is C01 (i=0, k=1)
                # [2] is C10 (i=1, k=0)
                # [3] is C11 (i=1, k=1)
                # So index is [i*2+k]? Or [k*2+i]?
                # C_ik. w = k*n + i = k*2 + i.
                # My list order: C00, C01, C10, C11.
                # C00: i=0, k=0. w=0.
                # C01: i=0, k=1. w=2.
                # C10: i=1, k=0. w=1.
                # C11: i=1, k=1. w=3.
                # So W_strassen[i*2+k] is NOT correct if list is [C00, C01, C10, C11].
                # List index 0: C00 (i=0, k=0)
                # List index 1: C01 (i=0, k=1)
                # List index 2: C10 (i=1, k=0)
                # List index 3: C11 (i=1, k=1)
                # So list index = i*2 + k?
                # i=0, k=0 -> 0. Correct.
                # i=0, k=1 -> 1. Correct.
                # i=1, k=0 -> 2. Correct.
                # i=1, k=1 -> 3. Correct.
                # So W_strassen[i*2+k] is the coefficient for C_ik.
                # But w = k*2 + i.
                # So we need to map correctly.
                
    # Let's redo W assignment carefully.
    # W_strassen[i][k] where i is row, k is col.
    # My list W_strassen is:
    # Row 0: [1, 0, 0, 1, -1, 0, 1] -> Coeffs for C00
    # Row 1: [0, 0, 1, 0, 1, 0, 0] -> Coeffs for C01
    # Row 2: [0, 1, 0, 1, 0, 0, 0] -> Coeffs for C10
    # Row 3: [1, -1, 1, 0, 0, 1, 0] -> Coeffs for C11
    # So W_strassen[i][r] is coeff for C_i, k? No, C_ik.
    # The list is indexed by [i*2+k]?
    # C00 (i=0, k=0) -> index 0.
    # C01 (i=0, k=1) -> index 1.
    # C10 (i=1, k=0) -> index 2.
    # C11 (i=1, k=1) -> index 3.
    # So W_strassen[i*2+k] is the vector of coeffs for C_ik.
    
    # In the loop:
    # for k in range(2):
    #   for i in range(2):
    #     w = k*2 + i
    #     coeff = W_strassen[i*2+k]
    #     W[w, r] += coeff[r]
    
    # I will fix the code below.
    
    # Reset U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: A1, B11, C11 (cols 0-1)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u = i * 3 + j
                U[u, r] += U_strassen[i][j]
        for j in range(2):
            for k in range(2):
                v = j * 4 + k
                V[v, r] += V_strassen[j][k]
        for k in range(2):
            for i in range(2):
                w = k * 2 + i
                # W_strassen index is i*2+k
                W[w, r] += W_strassen[i*2+k][r]
                
    # Block 2: A1, B12 (cols 2-3), C12 (cols 2-3)
    # Global k = 2 + k_loc
    for r in range(7, 14):
        r_loc = r - 7
        for i in range(2):
            for j in range(2):
                u = i * 3 + j
                U[u, r] += U_strassen[i][j]
        for j in range(2):
            for k_loc in range(2):
                k = 2 + k_loc
                v = j * 4 + k
                V[v, r] += V_strassen[j][k_loc]
        for k_loc in range(2):
            for i in range(2):
                k = 2 + k_loc
                w = k * 2 + i
                W[w, r] += W_strassen[i*2+k_loc][r_loc]
                
    # Block 3: A2 (col 2), B2 (row 2), C2 (all cols)
    # Rank 1
    r = 14
    for i in range(2):
        u = i * 3 + 2
        U[u, r] = 1
    for k in range(4):
        v = 2 * 4 + k
        V[v, r] = 1
    for k in range(4):
        for i in range(2):
            w = k * 2 + i
            W[w, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
