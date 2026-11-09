import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 8
    
    # Rank 56 decomposition using block Strassen
    # A (2x4) = [A0 | A1] where A0, A1 are 2x2
    # B (4x8) = [B0; B1] where B0, B1 are 4x8? No, B is 4x8.
    # Split B into two 2x8 blocks?
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x8.
    # C = A0 @ B_top + A1 @ B_bottom.
    # Each term is 2x2 @ 2x8.
    # Rank of 2x2 @ 2x8 is 28 (4 blocks of 2x2@2x2, each rank 7).
    # Total rank 56.

    # Strassen matrices for 2x2 @ 2x2
    # M1 = (A11) * (B22 - B11)
    # M2 = (A10 + A11) * B11
    # M3 = (A00 - A10) * B22
    # M4 = A00 * (B11 + B22)
    # M5 = (A00 + A11) * (B22 + B11)
    # M6 = (A11 - A01) * (B12 + B22)
    # M7 = (A01 - A10) * (B11 + B12)
    
    # We need to map these to the 2x2 @ 2x8 case.
    # B is 2x8, split into 4 blocks of 2x2: B0, B1, B2, B3.
    # For each block, we apply the 7 Strassen products.
    # So 4 * 7 = 28 terms per 2x2 @ 2x8.
    # Two such terms (for A0 and A1) -> 56 terms.

    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen coefficients for A (2x2)
    # Rows of x correspond to the 7 products. Cols to A elements [A00, A01, A10, A11]
    x = np.array([
        [0, 0, 0, 1], # A11
        [0, 0, 1, 1], # A10+A11
        [1, 0, -1, 0], # A00-A10
        [1, 0, 0, 0], # A00
        [1, 0, 0, 1], # A00+A11
        [0, 1, 0, -1], # A11-A01 (Wait, M6 is (A11-A01)) -> [0, -1, 0, 1]
        [0, 1, -1, 0] # A01-A10
    ], dtype=float)
    # Correction for M6: (A11 - A01) -> [0, -1, 0, 1]
    # Correction for M7: (A01 - A10) -> [0, 1, -1, 0]
    x[5] = [0, -1, 0, 1]
    x[6] = [0, 1, -1, 0]
    
    # Strassen coefficients for B (2x2)
    # Cols of y correspond to B elements [B11, B12, B21, B22] (Row-major 2x2)
    # M1: B22 - B11 -> [-1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    # M3: B22 -> [0, 0, 0, 1]
    # M4: B11 + B22 -> [1, 0, 0, 1]
    # M5: B22 + B11 -> [1, 0, 0, 1]
    # M6: B12 + B22 -> [0, 1, 0, 1]
    # M7: B11 + B12 -> [1, 1, 0, 0]
    
    y = np.array([
        [-1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ], dtype=float)
    
    # Strassen coefficients for C (2x2)
    # Rows of z correspond to C elements [C00, C01, C10, C11]
    # C00 = M4 - M5 + M3 + M7
    # C01 = M1 + M5
    # C10 = M2 + M4
    # C11 = M5 - M1 + M6 + M3
    # Order of M: 1..7
    z = np.array([
        [0, 0, 1, 1, -1, 0, 1], # C00
        [1, 0, 0, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, 0, 1, 0, -1, 1, 0] # C11 (M5 - M1 + M6 + M3) -> [-1, 0, 1, 0, 1, 1, 0]
    ], dtype=float)
    # Check C11: M5 - M1 + M6 + M3.
    # M1=0, M2=0, M3=1, M4=0, M5=-1, M6=1, M7=0.
    z[3] = [-1, 0, 1, 0, 1, 1, 0]
    
    r = 0
    # A0 part (cols 0,1 of A)
    for block in range(4): # 4 blocks of 2x2 in B_top (2x8)
        for s in range(7):
            # U: A0 elements
            # A0 is rows 0,1 cols 0,1 of A.
            # Index in U: i*m + j. i in 0,1. j in 0,1.
            for i in range(2):
                for j in range(2):
                    U[i*m + j, r] = x[s, i*2 + j]
            
            # V: B_top elements
            # B_top is rows 0,1 of B. Cols 2*block, 2*block+1.
            # Index in V: (i+m)*p + k? No.
            # V is m*p. m=4, p=8.
            # B is 4x8.
            # B_top is rows 0,1.
            # Block is 2x2.
            # Row j_B in 0,1. Col k_B in 2*block, 2*block+1.
            # V index: j_B * p + k_B.
            for j_B in range(2):
                for k_B in range(2):
                    V[j_B * p + (2*block + k_B), r] = y[s, j_B*2 + k_B]
            
            # W: C elements
            # C is 2x8.
            # C part is cols 2*block, 2*block+1.
            # Index in W: k_C * n + i_C.
            # k_C in 2*block, 2*block+1. i_C in 0,1.
            for i_C in range(2):
                for k_C in range(2):
                    W[(2*block + k_C) * n + i_C, r] = z[i_C*2 + k_C, s]
            
            r += 1
            
    # A1 part (cols 2,3 of A)
    for block in range(4): # 4 blocks of 2x2 in B_bottom (2x8)
        for s in range(7):
            # U: A1 elements
            # A1 is rows 0,1 cols 2,3 of A.
            for i in range(2):
                for j in range(2):
                    U[i*m + (2 + j), r] = x[s, i*2 + j]
            
            # V: B_bottom elements
            # B_bottom is rows 2,3 of B.
            for j_B in range(2):
                for k_B in range(2):
                    V[(2 + j_B) * p + (2*block + k_B), r] = y[s, j_B*2 + k_B]
            
            # W: C elements
            # Same C cols.
            for i_C in range(2):
                for k_C in range(2):
                    W[(2*block + k_C) * n + i_C, r] = z[i_C*2 + k_C, s]
            
            r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
