import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Coeffs for A: list of 7 lists of 4 coeffs (A00, A01, A10, A11)
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Coeffs for B: list of 7 lists of 4 coeffs (B00, B01, B10, B11)
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coeffs for C: list of 7 lists of 4 coeffs (C00, C01, C10, C11)
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    def add_strassen_block(k_offset):
        nonlocal col
        # Local A indices (0,0)..(1,1) map to global i in {0,1}, j in {0,1}
        # Local B indices (0,0)..(1,1) map to global i in {0,1}, j in {0,1}, k in {k_offset, k_offset+1}
        # Local C indices (0,0)..(1,1) map to global i in {0,1}, k in {k_offset, k_offset+1}
        
        for r in range(7):
            # U entries
            for li in range(2):
                for lj in range(2):
                    coeff = A_coeffs[r][li*2 + lj]
                    if coeff != 0:
                        gi = li
                        gj = lj
                        u_row = gi * m + gj
                        U[u_row][col] += coeff
            
            # V entries
            for li in range(2):
                for lk in range(2):
                    coeff = B_coeffs[r][li*2 + lk]
                    if coeff != 0:
                        gi = li
                        gj = lj # Wait, B index is (j, k). Local li is j, lk is k.
                        # In Strassen, B is 2x2. Rows are j, Cols are k.
                        # So local row index is j, local col index is k.
                        # B_coeffs[r][li*2 + lk] corresponds to B_{li, lk}.
                        # Global j = li, Global k = k_offset + lk.
                        v_row = gi * p + (k_offset + lk)
                        V[v_row][col] += coeff
                        
            # W entries
            for li in range(2):
                for lk in range(2):
                    coeff = C_coeffs[r][li*2 + lk]
                    if coeff != 0:
                        gi = li
                        gk = k_offset + lk
                        w_row = gk * n + gi
                        W[w_row][col] += coeff
                        
            col += 1

    # Block 1a: k in {0, 1}
    add_strassen_block(0)
    # Block 1b: k in {2, 3}
    add_strassen_block(2)
    # Block 2a: k in {4, 5}
    add_strassen_block(4)
    # Block 2b: k in {6, 7}
    add_strassen_block(6)
    
    # Term 3: A[0..1, 0..1] * B[0..1, 8]
    # A_{i,j} B_{j,8} -> C_{i,8}
    for i in range(2):
        for j in range(2):
            u_row = i * m + j
            v_row = j * p + 8
            w_row = 8 * n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1
            
    # Term 4: A[0..1, 2] * B[2, 0..8]
    # A_{i,2} B_{2,k} -> C_{i,k}
    for i in range(2):
        for k in range(9):
            u_row = i * m + 2
            v_row = 2 * p + k
            w_row = k * n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
