import time
import sys

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (12, rank)
    
    # We use a decomposition based on 6 Strassen 2x2x2 instances.
    # Left part: A_L (2x2) * B_T (2x6) split into 3 (2x2) blocks.
    # Right part: A_R (2x2) * B_B (2x6) split into 3 (2x2) blocks.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 multiplications coefficients
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Define the 7 terms for one 2x2x2 Strassen block
    # Each term is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of 4 values for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_coeffs: list of 4 values for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_coeffs: list of 4 values for C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_terms = [
        # P1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # P3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # P4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # P6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # P7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
    ]
    
    col = 0
    # 6 blocks: 3 for Left (A_L * B_T), 3 for Right (A_R * B_B)
    for side in range(2): # 0: Left, 1: Right
        for block_idx in range(3): # 0, 1, 2
            # Determine global offsets
            if side == 0:
                # A_L: cols 0, 1 of A
                a_col_offset = 0
                # B_T: rows 0, 1 of B
                b_row_offset = 0
            else:
                # A_R: cols 2, 3 of A
                a_col_offset = 2
                # B_B: rows 2, 3 of B
                b_row_offset = 2
            
            # B block cols: block_idx * 2
            b_col_offset = block_idx * 2
            
            for u_c, v_c, w_c in strassen_terms:
                # u_c has 4 elements: [A00, A01, A10, A11]
                # Map to global U indices: i * m + j
                # i is 0 or 1. j is a_col_offset + 0 or 1.
                for i in range(2):
                    for j_local in range(2):
                        val = u_c[i * 2 + j_local]
                        if val != 0:
                            j_global = a_col_offset + j_local
                            u_idx = i * m + j_global
                            U[u_idx][col] += val
                
                # v_c has 4 elements: [B00, B01, B10, B11]
                # Map to global V indices: j * p + k
                # j is b_row_offset + 0 or 1. k is b_col_offset + 0 or 1.
                for j_local in range(2):
                    for k_local in range(2):
                        val = v_c[j_local * 2 + k_local]
                        if val != 0:
                            j_global = b_row_offset + j_local
                            k_global = b_col_offset + k_local
                            v_idx = j_global * p + k_global
                            V[v_idx][col] += val
                
                # w_c has 4 elements: [C00, C01, C10, C11]
                # C is 2x6. i is 0 or 1. k is b_col_offset + 0 or 1.
                # W index: k * n + i
                for i in range(2):
                    for k_local in range(2):
                        val = w_c[i * 2 + k_local] # Note: w_c is indexed [i][k_local] in 2x2?
                        # My strassen_terms w_c is [C00, C01, C10, C11]
                        # So index i*2 + k_local corresponds to C[i][k_local]
                        if val != 0:
                            k_global = b_col_offset + k_local
                            w_idx = k_global * n + i
                            W[w_idx][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
