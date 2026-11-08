import time

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 7-term coefficients for 2x2 matrix multiplication
    # U_coeffs[s]: dict mapping (i_local, j_local) -> coeff for A block
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: (A00 + A11)
        {(1, 0): 1, (1, 1): 1},       # P2: (A10 + A11)
        {(0, 0): 1},                   # P3: A00
        {(1, 1): 1},                   # P4: A11
        {(0, 0): 1, (0, 1): 1},       # P5: (A00 + A01)
        {(1, 0): 1, (0, 0): -1},      # P6: (A10 - A00)
        {(0, 1): 1, (1, 1): -1}       # P7: (A01 - A11)
    ]
    
    # V_coeffs[s]: dict mapping (j_local, k_local) -> coeff for B block
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: (B00 + B11)
        {(0, 0): 1},                   # P2: B00
        {(0, 1): 1, (1, 1): -1},      # P3: (B01 - B11)
        {(1, 0): 1, (0, 0): -1},      # P4: (B10 - B00)
        {(1, 1): 1},                   # P5: B11
        {(0, 0): 1, (0, 1): 1},       # P6: (B00 + B01)
        {(1, 0): 1, (1, 1): 1}        # P7: (B10 + B11)
    ]
    
    # W_coeffs[s]: dict mapping (i_local, k_local) -> coeff for output C block
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: C00 += 1, C11 += 1
        {(1, 0): 1, (1, 1): -1},      # P2: C10 += 1, C11 += -1
        {(0, 1): 1, (1, 1): 1},       # P3: C01 += 1, C11 += 1
        {(0, 0): 1, (1, 0): 1},       # P4: C00 += 1, C10 += 1
        {(0, 0): -1, (0, 1): 1},      # P5: C00 += -1, C01 += 1
        {(1, 1): 1},                   # P6: C11 += 1
        {(0, 0): 1}                    # P7: C00 += 1
    ]
    
    col = 0
    for a in range(2):  # A column block: j in {0,1} or {2,3}
        for b in range(3):  # B column block: k in {0,1}, {2,3}, or {4,5}
            for s in range(7):  # Strassen term
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i][col] += coeff
                
                col += 1
                
                if time.time() > deadline - 5:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
