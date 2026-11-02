import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Rank-42 decomposition using Strassen algorithm on 2x2 blocks
    # A is 2x4, B is 4x6
    # Split A into A_L (cols 0,1) and A_R (cols 2,3)
    # Split B into B_T (rows 0,1) and B_B (rows 2,3)
    # C = A_L @ B_T + A_R @ B_B
    # B_T is 2x6, split into 3 blocks of 2x2: BT_0, BT_1, BT_2
    # B_B is 2x6, split into 3 blocks of 2x2: BB_0, BB_1, BB_2
    # Total 6 blocks of 2x2 matmul.
    # Each 2x2 matmul has rank 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 matmul C = A @ B
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
    
    # Local indices for A: (a, b) where a=row, b=col (0..1)
    # Local indices for B: (c, d) where c=row, d=col (0..1)
    # Local indices for C: (e, f) where e=row, f=col (0..1)
    
    # Strassen terms definition:
    # Each term is a list of (local_A_idx, coeff), (local_B_idx, coeff), (local_C_idx, coeff)
    # We need to add contributions to U, V, W.
    # U corresponds to A factors. V to B factors. W to C factors.
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # Define the 7 Strassen multipliers
    # For each multiplier r in 0..6:
    # A_part: linear combo of A entries
    # B_part: linear combo of B entries
    # C_part: linear combo of C entries
    
    # We will iterate over the 6 blocks.
    # Block 0: A_L, B_T[0:2] (cols 0,1 of B_T)
    # Block 1: A_L, B_T[2:4] (cols 2,3 of B_T)
    # Block 2: A_L, B_T[4:6] (cols 4,5 of B_T)
    # Block 3: A_R, B_B[0:2] (cols 0,1 of B_B)
    # Block 4: A_R, B_B[2:4] (cols 2,3 of B_B)
    # Block 5: A_R, B_B[4:6] (cols 4,5 of B_B)
    
    col = 0
    for block_id in range(6):
        # Determine if using A_L or A_R
        use_A_L = (block_id < 3)
        # Determine which column block of the 2x6 B part
        b_col_block = block_id % 3
        
        for r in range(7):
            # Define coefficients for A, B, C for this Strassen term
            # A coefficients: dict {(a,b): coeff}
            # B coefficients: dict {(c,d): coeff}
            # C coefficients: dict {(e,f): coeff}
            
            A_coeffs = {}
            B_coeffs = {}
            C_coeffs = {}
            
            if r == 0: # M1
                A_coeffs[(0,0)] = 1; A_coeffs[(1,1)] = 1
                B_coeffs[(0,0)] = 1; B_coeffs[(1,1)] = 1
                C_coeffs[(0,0)] = 1; C_coeffs[(1,1)] = 1
            elif r == 1: # M2
                A_coeffs[(1,0)] = 1; A_coeffs[(1,1)] = 1
                B_coeffs[(0,0)] = 1
                C_coeffs[(1,0)] = 1; C_coeffs[(1,1)] = -1
            elif r == 2: # M3
                A_coeffs[(0,0)] = 1
                B_coeffs[(0,1)] = 1; B_coeffs[(1,1)] = -1
                C_coeffs[(0,1)] = 1; C_coeffs[(1,1)] = 1
            elif r == 3: # M4
                A_coeffs[(1,1)] = 1
                B_coeffs[(1,0)] = 1; B_coeffs[(0,0)] = -1
                C_coeffs[(0,0)] = 1; C_coeffs[(1,0)] = 1
            elif r == 4: # M5
                A_coeffs[(0,0)] = 1; A_coeffs[(0,1)] = 1
                B_coeffs[(1,1)] = 1
                C_coeffs[(0,0)] = -1; C_coeffs[(0,1)] = 1
            elif r == 5: # M6
                A_coeffs[(1,0)] = 1; A_coeffs[(0,0)] = -1
                B_coeffs[(0,0)] = 1; B_coeffs[(0,1)] = 1
                C_coeffs[(1,1)] = 1
            elif r == 6: # M7
                A_coeffs[(0,1)] = 1; A_coeffs[(1,1)] = -1
                B_coeffs[(1,0)] = 1; B_coeffs[(1,1)] = 1
                C_coeffs[(0,0)] = 1
            
            # Apply to global indices
            for (a, b), coeff_a in A_coeffs.items():
                # a is local row (0,1), b is local col (0,1)
                i = a
                if use_A_L:
                    j = b
                else:
                    j = 2 + b
                U[i * m + j, col] += coeff_a
                
            for (c, d), coeff_b in B_coeffs.items():
                # c is local row (0,1), d is local col (0,1)
                if use_A_L:
                    # B_T part
                    j = c
                else:
                    # B_B part
                    j = 2 + c
                k = 2 * b_col_block + d
                V[j * p + k, col] += coeff_b
                
            for (e, f), coeff_c in C_coeffs.items():
                # e is local row (0,1), f is local col (0,1)
                i = e
                k = 2 * b_col_block + f
                W[k * n + i, col] += coeff_c
                
            col += 1
            
            if time.time() > deadline - 5.0:
                # Safety break, though 42 is small
                break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
