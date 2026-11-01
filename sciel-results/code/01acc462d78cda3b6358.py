"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import fractions
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen's algorithm for 2x2 blocks.
    # Decompose the 2x4 x 4x6 multiplication into 6 block multiplications of 2x2 matrices.
    # A (2x4) = [A0, A1] where A0, A1 are 2x2.
    # B (4x6) = [B00 B01 B02; B10 B11 B12] where each is 2x2.
    # C (2x6) = [A0*B00 + A1*B10, A0*B01 + A1*B11, A0*B02 + A1*B12]
    
    # Each 2x2 multiplication uses rank 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will build U, V, W matrices of size (2*4)x42, (4*6)x42, (6*2)x42.
    # Wait, W shape is (p*n, rank) = (6*2, 42) = (12, 42).
    
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Define the 6 block multiplications:
    # (A_block, B_block, C_block)
    # A is 2x4. Rows 0,1. Cols 0,1 (A0), 2,3 (A1).
    # B is 4x6. Rows 0,1 (top), 2,3 (bottom). Cols 0,1 (col0), 2,3 (col1), 4,5 (col2).
    # C is 2x6. Rows 0,1. Cols 0,1 (col0), 2,3 (col1), 4,5 (col2).
    
    # Block A0: rows 0-1, cols 0-1 of A.
    # Block A1: rows 0-1, cols 2-3 of A.
    
    # Block B00: rows 0-1, cols 0-1 of B.
    # Block B10: rows 2-3, cols 0-1 of B.
    # Block B01: rows 0-1, cols 2-3 of B.
    # Block B11: rows 2-3, cols 2-3 of B.
    # Block B02: rows 0-1, cols 4-5 of B.
    # Block B12: rows 2-3, cols 4-5 of B.
    
    # Terms in C:
    # C_col0 = A0*B00 + A1*B10
    # C_col1 = A0*B01 + A1*B11
    # C_col2 = A0*B02 + A1*B12
    
    # We define a helper to add a Strassen decomposition for a specific 2x2x2 multiplication.
    # The Strassen decomposition is defined on local indices 0..1 for rows/cols.
    # We map local indices to global indices.
    
    def add_strassen_term(U, V, W, offset, 
                          a_row_start, a_col_start, 
                          b_row_start, b_col_start, 
                          c_row_start, c_col_start):
        """
        Adds the 7 rank-1 tensors for a 2x2x2 multiplication.
        a_row_start, a_col_start: top-left corner of A block (2x2) in global A (2x4)
        b_row_start, b_col_start: top-left corner of B block (2x2) in global B (4x6)
        c_row_start, c_col_start: top-left corner of C block (2x2) in global C (2x6)
        """
        # Strassen coefficients:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Local indices for A: 00, 01, 10, 11
        # Global A index: (a_row_start + i) * m + (a_col_start + j)
        # Local indices for B: 00, 01, 10, 11
        # Global B index: (b_row_start + i) * p + (b_col_start + j)
        # Local indices for C: 00, 01, 10, 11
        # Global C index: (c_row_start + i) * n + (c_col_start + j)  -- Note: C is p*n? No, C is n*p?
        # Task says: W[k*n+i, r]. So C index is k*n + i where k is col, i is row?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Usually C[i,k] = sum A[i,j]B[j,k].
        # If W index is k*n+i, that means index = col * n + row.
        # Let's check the seed program's verify:
        # c = k * n + i.
        # a = i * m + j.
        # b = j * p + k.
        # So A[i,j], B[j,k], C[i,k] (stored at index k*n+i).
        
        # Strassen contributions to C:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the 7 terms as tuples (coeff_A, coeff_B, coeff_C) for each local cell.
        # Actually, it's easier to define the 7 rank-1 decompositions directly.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # A factors: A00 gets 1, A11 gets 1. Others 0.
        # B factors: B00 gets 1, B11 gets 1. Others 0.
        # C factors: C00 gets 1, C11 gets 1.
        
        # Let's list the 7 terms with their A, B, C support and coefficients.
        # Each term is a set of (local_a, local_b, local_c) -> coeff.
        # But we need U, V, W vectors.
        
        # Term 1:
        # A: (0,0):1, (1,1):1
        # B: (0,0):1, (1,1):1
        # C: (0,0):1, (1,1):1
        
        # Term 2:
        # A: (1,0):1, (1,1):1
        # B: (0,0):1
        # C: (1,0):1
        
        # Term 3:
        # A: (0,0):1
        # B: (0,1):1, (1,1):-1
        # C: (0,1):1
        
        # Term 4:
        # A: (1,1):1
        # B: (1,0):1, (0,0):-1
        # C: (1,0):1, (0,0):-1  <-- Wait, C10 = M2 + M4, C00 = M1 + M4 - M5 + M7.
        # So M4 contributes +1 to C10 and +1 to C00?
        # C00 = M1 + M4 - M5 + M7. Yes, +1.
        # C10 = M2 + M4. Yes, +1.
        
        # Term 5:
        # A: (0,0):1, (0,1):1
        # B: (1,1):1
        # C: (0,0):-1, (0,1):1
        
        # Term 6:
        # A: (1,0):1, (0,0):-1
        # B: (0,0):1, (0,1):1
        # C: (1,1):1
        
        # Term 7:
        # A: (0,1):1, (1,1):-1
        # B: (1,0):1, (1,1):1
        # C: (0,0):1, (1,1):-1
        
        # Let's verify C00: M1(1) + M4(1) + M5(-1) + M7(1) = 2 - 1 + 1 = 2? No.
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # M1 = (A00+A11)(B00+B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M7 = (A01-A11)(B10+B11)
        
        # Let's check coefficients for C00:
        # M1: +1
        # M4: +1
        # M5: -1
        # M7: +1
        
        # My Term 5 C coeff was -1 for C00. Correct.
        # My Term 7 C coeff was +1 for C00. Correct.
        # My Term 4 C coeff was +1 for C00. Correct.
        # My Term 1 C coeff was +1 for C00. Correct.
        
        # So for each term r in 0..6:
        # We update U, V, W at the global indices corresponding to the non-zero local indices.
        
        terms = [
            # (A_dict, B_dict, C_dict)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # M2
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # M3
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}), # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}) # M7
        ]
        
        for r in range(7):
            global_r = offset + r
            a_dict, b_dict, c_dict = terms[r]
            
            for (i, j), val in a_dict.items():
                global_a = (a_row_start + i) * m + (a_col_start + j)
                U[global_a][global_r] += val
                
            for (i, j), val in b_dict.items():
                global_b = (b_row_start + i) * p + (b_col_start + j)
                V[global_b][global_r] += val
                
            for (i, j), val in c_dict.items():
                # C index: k*n + i. Here i is row, j is col.
                # So index = j * n + i.
                global_c = (c_col_start + j) * n + (c_row_start + i)
                W[global_c][global_r] += val

    # Now, instantiate the 6 block multiplications.
    
    # 1. A0 * B00 -> C_col0 (part 1)
    # A0: rows 0-1, cols 0-1. a_row_start=0, a_col_start=0.
    # B00: rows 0-1, cols 0-1. b_row_start=0, b_col_start=0.
    # C_col0: rows 0-1, cols 0-1. c_row_start=0, c_col_start=0.
    add_strassen_term(U, V, W, 0, 0, 0, 0, 0, 0, 0)
    
    # 2. A1 * B10 -> C_col0 (part 2)
    # A1: rows 0-1, cols 2-3. a_row_start=0, a_col_start=2.
    # B10: rows 2-3, cols 0-1. b_row_start=2, b_col_start=0.
    # C_col0: rows 0-1, cols 0-1. c_row_start=0, c_col_start=0.
    add_strassen_term(U, V, W, 7, 0, 2, 2, 0, 0, 0)
    
    # 3. A0 * B01 -> C_col1 (part 1)
    # A0: rows 0-1, cols 0-1. a_row_start=0, a_col_start=0.
    # B01: rows 0-1, cols 2-3. b_row_start=0, b_col_start=2.
    # C_col1: rows 0-1, cols 2-3. c_row_start=0, c_col_start=2.
    add_strassen_term(U, V, W, 14, 0, 0, 0, 2, 0, 2)
    
    # 4. A1 * B11 -> C_col1 (part 2)
    # A1: rows 0-1, cols 2-3. a_row_start=0, a_col_start=2.
    # B11: rows 2-3, cols 2-3. b_row_start=2, b_col_start=2.
    # C_col1: rows 0-1, cols 2-3. c_row_start=0, c_col_start=2.
    add_strassen_term(U, V, W, 21, 0, 2, 2, 2, 0, 2)
    
    # 5. A0 * B02 -> C_col2 (part 1)
    # A0: rows 0-1, cols 0-1. a_row_start=0, a_col_start=0.
    # B02: rows 0-1, cols 4-5. b_row_start=0, b_col_start=4.
    # C_col2: rows 0-1, cols 4-5. c_row_start=0, c_col_start=4.
    add_strassen_term(U, V, W, 28, 0, 0, 0, 4, 0, 4)
    
    # 6. A1 * B12 -> C_col2 (part 2)
    # A1: rows 0-1, cols 2-3. a_row_start=0, a_col_start=2.
    # B12: rows 2-3, cols 4-5. b_row_start=2, b_col_start=4.
    # C_col2: rows 0-1, cols 4-5. c_row_start=0, c_col_start=4.
    add_strassen_term(U, V, W, 35, 0, 2, 2, 4, 0, 4)
    
    # Verification
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U_s = [[0] * rank for _ in range(n * m)]
        V_s = [[0] * rank for _ in range(m * p)]
        W_s = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
