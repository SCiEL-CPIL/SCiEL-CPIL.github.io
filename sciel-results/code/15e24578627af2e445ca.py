import numpy as np

def construct():
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        # M2: (A10+A11)B00 -> C10+=1, C11-=1
        # M3: A00(B01-B11) -> C01+=1, C11+=1
        # M4: A11(B10-B00) -> C00+=1, C10+=1
        # M5: (A00+A01)B11 -> C00-=1, C01+=1
        # M6: (A10-A00)(B00+B01) -> C11+=1
        # M7: (A01-A11)(B10+B11) -> C00+=1, C01-=1
        
        # Coefficients for U (A)
        U_coeffs = [
            [1, 1, 0, 0], # M1: A00, A11
            [0, 0, 1, 1], # M2: A10, A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00, A01
            [-1, 0, 1, 0],# M6: A00, A10
            [0, 1, 0, -1] # M7: A01, A11
        ]
        # Coefficients for V (B)
        V_coeffs = [
            [1, 0, 0, 1], # M1: B00, B11
            [1, 0, 0, 0], # M2: B00
            [0, 1, 0, -1],# M3: B01, B11
            [0, 0, 1, -1],# M4: B10, B00
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00, B01
            [0, 0, 1, 1]  # M7: B10, B11
        ]
        # Coefficients for W (C)
        # C00, C01, C10, C11
        W_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1],# M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, -1, 0, 0] # M7
        ]
        
        for s in range(7):
            # Fill U
            for d in range(4):
                if U_coeffs[s][d] != 0:
                    U[a_idx[d]][col] = U_coeffs[s][d]
            # Fill V
            for d in range(4):
                if V_coeffs[s][d] != 0:
                    V[b_idx[d]][col] = V_coeffs[s][d]
            # Fill W
            for d in range(4):
                if W_coeffs[s][d] != 0:
                    W[c_idx[d]][col] = W_coeffs[s][d]
            col += 1

    # Define index mappings
    # A(1): cols 0,1
    a1 = [0, 1, 5, 6] # A00, A01, A10, A11
    # A(2): cols 2,3
    a2 = [2, 3, 7, 8]
    
    # B(1): rows 0,1
    # B(1)_1: cols 0,1
    b11 = [0, 1, 6, 7]
    # B(1)_2: cols 2,3
    b12 = [2, 3, 8, 9]
    # B(1)_3: cols 4,5
    b13 = [4, 5, 10, 11]
    
    # B(2): rows 2,3
    # B(2)_1: cols 0,1
    b21 = [12, 13, 18, 19]
    # B(2)_2: cols 2,3
    b22 = [14, 15, 20, 21]
    # B(2)_3: cols 4,5
    b23 = [16, 17, 22, 23]
    
    # C indices: k*n + i
    # Block 1 (k=0,1)
    c1 = [0, 1, 2, 3] # C00, C01, C10, C11
    # Block 2 (k=2,3)
    c2 = [4, 6, 5, 7]
    # Block 3 (k=4,5)
    c3 = [8, 10, 9, 11]
    
    # Add 6 Strassen blocks
    add_strassen(a1, b11, c1)
    add_strassen(a1, b12, c2)
    add_strassen(a1, b13, c3)
    
    add_strassen(a2, b21, c1)
    add_strassen(a2, b22, c2)
    add_strassen(a2, b23, c3)
    
    # Add outer product terms for A(3)B(3)
    # A(3) is col 4 of A. Indices 4 (row 0), 9 (row 1)
    # B(3) is row 4 of B. Indices 24..29
    for k in range(6):
        # Term for C[0,k]
        U[4][col] = 1
        V[24 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # Term for C[1,k]
        U[9][col] = 1
        V[24 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
