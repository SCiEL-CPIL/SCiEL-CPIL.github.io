import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We'll decompose C = A * B where A is 4x4, B is 4x6
    # Split B into B1 (4x4) and B2 (4x2)
    # C1 = A * B1 (4x4) using Strassen rank 7
    # C2 = A * B2 (4x2) using schoolbook rank 32
    # Total rank R = 7 + 32 = 39
    
    R = 39
    U = np.zeros((n * m, R))  # (16, 39)
    V = np.zeros((m * p, R))  # (24, 39)
    W = np.zeros((p * n, R))  # (24, 39)
    
    # --- Part 1: Strassen for 4x4 x 4x4 (columns 0-3 of B) ---
    # Standard Strassen for 2x2 blocks
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # Strassen intermediate matrices:
    # M1 = (A10 + A11) * (B00 + B01)
    # M2 = (A00 + A10) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (A_coeff, B_coeff, C_blocks)
        # M1 = (A10 + A11) * (B00 + B01)
        ({'A10': 1, 'A11': 1}, {'B00': 1, 'B01': 1}, {'C00': 1, 'C11': 1}),
        # M2 = (A00 + A10) * B00
        ({'A00': 1, 'A10': 1}, {'B00': 1}, {'C10': 1, 'C11': -1}),
        # M3 = A00 * (B01 - B11)
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': 1}),
        # M4 = A11 * (B10 - B00)
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        # M5 = (A00 + A01) * B11
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        # M6 = (A10 - A00) * (B00 + B10)
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B10': 1}, {'C11': 1}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1}),
    ]
    
    col = 0
    for A_coeff, B_coeff, C_coeff in strassen_terms:
        # U: contributions from A blocks
        for A_name, a_val in A_coeff.items():
            if A_name == 'A00':
                rows = [0, 1]
                cols = [0, 1]
            elif A_name == 'A01':
                rows = [0, 1]
                cols = [2, 3]
            elif A_name == 'A10':
                rows = [2, 3]
                cols = [0, 1]
            elif A_name == 'A11':
                rows = [2, 3]
                cols = [2, 3]
            for i in rows:
                for j in cols:
                    U[i * m + j, col] += a_val
        
        # V: contributions from B blocks (only cols 0-3)
        for B_name, b_val in B_coeff.items():
            if B_name == 'B00':
                brows = [0, 1]
                bcols = [0, 1]
            elif B_name == 'B01':
                brows = [0, 1]
                bcols = [2, 3]
            elif B_name == 'B10':
                brows = [2, 3]
                bcols = [0, 1]
            elif B_name == 'B11':
                brows = [2, 3]
                bcols = [2, 3]
            for j in brows:
                for k in bcols:
                    V[j * p + k, col] += b_val
        
        # W: contributions to C blocks
        for C_name, c_val in C_coeff.items():
            if C_name == 'C00':
                wrows = [0, 1]
                wcols = [0, 1]
            elif C_name == 'C01':
                wrows = [0, 1]
                wcols = [2, 3]
            elif C_name == 'C10':
                wrows = [2, 3]
                wcols = [0, 1]
            elif C_name == 'C11':
                wrows = [2, 3]
                wcols = [2, 3]
            for k in wcols:
                for i in wrows:
                    W[k * n + i, col] += c_val
        
        col += 1
    
    # --- Part 2: Schoolbook for 4x4 x 4x2 (columns 4-5 of B) ---
    # C2[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, k in 4,5, j in 0..3
    # 4 * 2 * 4 = 32 terms
    for i in range(4):
        for k in [4, 5]:
            for j in range(4):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
