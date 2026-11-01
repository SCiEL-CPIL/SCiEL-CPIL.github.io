import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 (Strassen) = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = A11 (B11 + B22)
    # M2 = (A11 + A12) B22
    # M3 = (A21 + A22) B11
    # M4 = A22 (B11 - B22)
    # M5 = (A11 + A21) B21
    # M6 = (A12 - A22) (B12 + B22)
    # M7 = (A11 - A21) (B11 + B12)
    
    # Indices in flattened A (row major): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices in flattened B (row major): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices in flattened C (row major): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Strassen U (A coefficients)
    # r=0: A11
    # r=1: A11+A12
    # r=2: A21+A22
    # r=3: A22
    # r=4: A11+A21
    # r=5: A12-A22
    # r=6: A11-A21
    
    U_strassen = [
        [1, 0, 0, 0], # A11
        [1, 1, 0, 0], # A11+A12
        [0, 0, 1, 1], # A21+A22
        [0, 0, 0, 1], # A22
        [1, 0, 1, 0], # A11+A21
        [0, 1, 0, -1],# A12-A22
        [1, 0, -1, 0] # A11-A21
    ]
    
    # Strassen V (B coefficients)
    # r=0: B11+B22
    # r=1: B22
    # r=2: B11
    # r=3: B11-B22
    # r=4: B21
    # r=5: B12+B22
    # r=6: B11+B12
    
    V_strassen = [
        [1, 0, 0, 1], # B11+B22
        [0, 0, 0, 1], # B22
        [1, 0, 0, 0], # B11
        [1, 0, 0, -1],# B11-B22
        [0, 0, 1, 0], # B21
        [0, 1, 0, 1], # B12+B22
        [1, 1, 0, 0]  # B11+B12
    ]
    
    # Strassen W (C coefficients)
    # C11 = M1 + M4 - M2 + M6
    # C12 = M3 + M5
    # C21 = M5 + M7
    # C22 = M1 - M3 + M4 + M7
    
    W_strassen = [
        [1, 0, 0, 1], # M1, M4, M2, M6 -> C11
        [0, 1, 1, 0], # M3, M5 -> C12
        [0, 1, 0, 1], # M5, M7 -> C21
        [1, 0, -1, 1] # M1, M3, M4, M7 -> C22
    ]
    # Note: W_strassen[r][c] is coeff of M_r in C_c?
    # My W_strassen above is:
    # Row 0 (C11): M1(1), M4(1), M2(-1), M6(1)
    # Let's align indices.
    # M1 is r=0. M2 is r=1. M3 is r=2. M4 is r=3. M5 is r=4. M6 is r=5. M7 is r=6.
    
    W_strassen_correct = [
        [1, -1, 0, 1, 0, 1, 0], # C11: M1 - M2 + M4 + M6
        [0, 0, 1, 0, 1, 0, 0],  # C12: M3 + M5
        [0, 0, 0, 0, 1, 0, 1],  # C21: M5 + M7
        [1, 0, -1, 1, 0, 0, 1]  # C22: M1 - M3 + M4 + M7
    ]
    
    # For each of the 4 blocks (q=0..3)
    for q in range(4):
        base_col = q * 7
        for r in range(7):
            col = base_col + r
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    idx_A = i * 2 + j
                    U[idx_A][col] = U_strassen[r][idx_A]
            
            # Fill V
            for j in range(2):
                for k in range(2):
                    # B index in block q: j * 2 + k
                    # Global B index: j * 16 + (q * 2 + k)
                    idx_B_local = j * 2 + k
                    idx_B_global = j * 16 + (q * 2 + k)
                    V[idx_B_global][col] = V_strassen[r][idx_B_local]
            
            # Fill W
            for k in range(2):
                for i in range(2):
                    # C index in block q: k * 2 + i
                    # Global C index: k * 2 + i ?
                    # Trace convention: W[k*n+i, r]
                    # n=2. So idx_C = k * 2 + i.
                    # But C is 2x16.
                    # The problem says W shape (p*n, rank).
                    # p=16, n=2.
                    # Index is k*n + i.
                    # k is column of C (0..15). i is row of C (0..1).
                    # So for block q, columns are 4q..4q+3.
                    # k_global = 4*q + k_local.
                    idx_C_local = k * 2 + i
                    idx_C_global = (4 * q + k) * 2 + i
                    W[idx_C_global][col] = W_strassen_correct[idx_C_local][r]
                    
    return {"U": U, "V": V, "W": W}
