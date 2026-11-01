import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    col = 0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # M1: (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2: (A10+A11)B00 -> C10+=M2, C11-=M2
    # M3: A00(B01-B11) -> C01+=M3, C11+=M3
    # M4: A11(B10-B00) -> C00+=M4, C10+=M4
    # M5: (A00+A01)B11 -> C00-=M5, C01+=M5
    # M6: (A10-A00)(B00+B01) -> C11+=M6
    # M7: (A01-A11)(B10+B11) -> C00+=M7
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # Coeffs are dicts mapping local index (0..3) to coefficient
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1},       {2: 1, 3: -1}),  # M2
        ({0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1}),  # M3
        ({3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1}),  # M4
        ({0: 1, 1: 1}, {3: 1},       {0: -1, 1: 1}),  # M5
        ({2: 1, 0: -1},{0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1},{2: 1, 3: 1}, {0: 1})          # M7
    ]
    
    # 8 blocks: 2 parts of A (cols 0-1, 2-3) x 4 blocks of B (cols 0-1, 2-3, 4-5, 6-7)
    # A is 2x4. B is 4x8. C is 2x8.
    # C[i, k] = sum_j A[i, j] B[j, k]
    
    for a_part in range(2):
        # a_part 0: A cols 0,1. a_part 1: A cols 2,3
        a_col_start = a_part * 2
        
        for b_block in range(4):
            # b_block 0: B cols 0,1. b_block 1: B cols 2,3, etc.
            b_col_start = b_block * 2
            
            # The rows of B involved are the same as the columns of A involved.
            # So B rows are a_col_start to a_col_start+1.
            b_row_start = a_col_start
            
            for a_idx, a_coeff in []: pass # placeholder, we iterate terms below
            
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                # Fill U (A factor)
                for local_idx, coeff in A_coeffs.items():
                    r, c = divmod(local_idx, 2)
                    global_i = r
                    global_j = a_col_start + c
                    global_u = global_i * m + global_j
                    U[global_u, col] += coeff
                
                # Fill V (B factor)
                for local_idx, coeff in B_coeffs.items():
                    r, c = divmod(local_idx, 2)
                    global_j = b_row_start + r
                    global_k = b_col_start + c
                    global_v = global_j * p + global_k
                    V[global_v, col] += coeff
                
                # Fill W (C factor)
                # C index is k*n + i
                for local_idx, coeff in C_coeffs.items():
                    r, c = divmod(local_idx, 2)
                    global_i = r
                    global_k = b_col_start + c
                    global_w = global_k * n + global_i
                    W[global_w, col] += coeff
                
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
