import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # We will construct a rank-65 decomposition:
    # 49 terms for the 4x4x4 part using Strassen
    # 16 terms for the 4x4x1 part (column 4 of B)
    
    U = np.zeros((n * m, 65), dtype=float)  # (16, 65)
    V = np.zeros((m * p, 65), dtype=float)  # (20, 65)
    W = np.zeros((p * n, 65), dtype=float)  # (20, 65)
    
    # Helper functions
    def a_idx(i, j):
        return i * m + j  # i in 0..3, j in 0..3
    
    def b_idx(j, k):
        return j * p + k  # j in 0..3, k in 0..4
    
    def c_idx(i, k):
        return k * n + i  # i in 0..3, k in 0..4
    
    # Part 1: 4x4x1 part (16 terms for column 4)
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[a_idx(i, j), col] = 1
            V[b_idx(j, 4), col] = 1
            W[c_idx(i, 4), col] = 1
    
    # Part 2: 4x4x4 part using Strassen (49 terms)
    # We'll use a recursive approach to generate the 49 terms
    
    # Strassen 2x2 formulas
    # For 2x2 matrices:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A10) * B11
    # M6 = (A11 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4, we split into 2x2 blocks:
    # A_blocks: A00, A01, A10, A11 (each 2x2)
    # B_blocks: B00, B01, B10, B11 (each 2x2)
    
    # First level: 7 block products
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A10) * B11
    # M6 = (A11 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B10)
    
    # Each block product is a 2x2 matrix product of linear combinations
    # We apply Strassen again to each 2x2 block product
    
    # Let's generate the 49 terms explicitly
    # Term index will be 0..48
    
    terms = []
    
    # Define the 7 block combinations for the first level
    # Each is (A_coeffs, B_coeffs) where coeffs are dicts {block_idx: coeff}
    block_combos = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),      # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),                  # M2
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),                 # M3
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}),                 # M4
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}),                  # M5
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),       # M6
        ({(0,1): 1, (1,1): -1}, {(1,1): 1, (1,0): 1})        # M7
    ]
    
    # For each block combo, we need to generate 7 scalar products
    # using Strassen on the 2x2 coefficients
    
    # Strassen 2x2 scalar products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A10)B11
    # P6 = (A11-A00)(B00+B10)
    # P7 = (A01-A11)(B11+B10)
    
    # Output mapping:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's define the 7 Strassen 2x2 patterns
    # Each pattern is (A_coeffs, B_coeffs, C_contrib)
    # A_coeffs and B_coeffs are dicts {(row,col): coeff} for 2x2
    # C_contrib is dict {(row,col): coeff} for 2x2 output
    
    strassen_2x2_patterns = [
        # P1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1, (1,1): 1}),
        # P2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # P3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1}),
        # P4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # P5
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # P6
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # P7
        ({(0,1): 1, (1,1): -1}, {(1,1): 1, (1,0): 1}, {(0,0): 1})
    ]
    
    # Now generate all 49 terms
    for bc_idx, (A_block_coeffs, B_block_coeffs) in enumerate(block_combos):
        for p_idx, (A_pat, B_pat, C_pat) in enumerate(strassen_2x2_patterns):
            term_idx = bc_idx * 7 + p_idx
            
            # Create U, V, W vectors for this term
            U_vec = np.zeros(16)
            V_vec = np.zeros(20)
            W_vec = np.zeros(20)
            
            # Expand A_block_coeffs and A_pat to get full 4x4 A coefficients
            # A_block_coeffs: {block_idx: coeff} where block_idx is (bi, bj) in 0..1
            # A_pat: {(i,j): coeff} where i,j in 0..1 (within each block)
            
            # For each block in A_block_coeffs and each position in A_pat:
            for (bi, bj), block_coeff in A_block_coeffs.items():
                for (ai, aj), pat_coeff in A_pat.items():
                    # Global position: row = 2*bi + ai, col = 2*bj + aj
                    row = 2 * bi + ai
                    col = 2 * bj + aj
                    U_vec[row * 4 + col] += block_coeff * pat_coeff
            
            # Expand B_block_coeffs and B_pat to get full 4x4 B coefficients
            # B is 4x5, but we only use columns 0..3 for the 4x4x4 part
            for (bi, bj), block_coeff in B_block_coeffs.items():
                for (ai, aj), pat_coeff in B_pat.items():
                    # Global position: row = 2*bi + ai, col = 2*bj + aj
                    row = 2 * bi + ai
                    col = 2 * bj + aj
                    V_vec[row * 5 + col] += block_coeff * pat_coeff
            
            # Expand C_pat to get full 4x4 C coefficients
            # C_pat: {(i,j): coeff} where i,j in 0..1
            # But we need to map this to the 2x2 block structure
            # The output of each block product M_k is a 2x2 matrix
            # We need to know which block of C this contributes to
            
            # For block combo bc_idx, the result contributes to specific blocks of C
            # Let's determine the C block mapping for each M_k
            
            # M1 -> C00, C01, C10, C11 (all blocks)
            # Actually, each M_k is a 2x2 matrix that contributes to the corresponding block
            
            # Let me reconsider: The 7 block products M1..M7 are each 2x2 matrices
            # M1 = (A00+A11)(B00+B11) -> 2x2 matrix
            # M2 = (A10+A11)B00 -> 2x2 matrix
            # etc.
            
            # The final C is:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for each M_k, we need to know which C block(s) it contributes to
            # and with what coefficient
            
            c_block_mapping = {
                0: [(0,0,1), (1,0,1), (1,1,1)],   # M1 -> C00(+), C10(+), C11(+)
                1: [(1,0,1), (1,1,-1)],           # M2 -> C10(+), C11(-)
                2: [(0,1,1), (1,1,1)],            # M3 -> C01(+), C11(+)
                3: [(0,0,1), (1,0,1)],            # M4 -> C00(+), C10(+)
                4: [(0,0,-1), (0,1,1)],           # M5 -> C00(-), C01(+)
                5: [(1,1,1)],                     # M6 -> C11(+)
                6: [(0,0,1)]                      # M7 -> C00(+)
            }
            
            # Get the C block contributions for this M_k
            for (ci, cj, coeff) in c_block_mapping[bc_idx]:
                # The 2x2 result from this term contributes to block (ci, cj) of C
                # For each position in C_pat:
                for (ri, rj), pat_coeff in C_pat.items():
                    # Global position: row = 2*ci + ri, col = 2*cj + rj
                    row = 2 * ci + ri
                    col = 2 * cj + rj
                    W_vec[col * 4 + row] += coeff * pat_coeff
            
            # Store the term
            U[:, term_idx] = U_vec
            V[:, term_idx] = V_vec
            W[:, term_idx] = W_vec
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
