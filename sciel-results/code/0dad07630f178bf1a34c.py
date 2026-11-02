"""Matrix multiplication 6x6x6 - rank-minimal tensor decomposition search."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use 2x2 block Strassen with 3x3 blocks
    # Each 3x3 multiplication uses schoolbook rank 27
    # Total rank = 7 * 27 = 189
    
    rank = 7 * 27  # 189
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen products: (A_block_coeffs, B_block_coeffs)
    # Block indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1 = (A11+A22)*(B11+B22)
        ([(2, 1), (3, 1)], [(0, 1)]),           # M2 = (A21+A22)*B11
        ([(0, 1)], [(1, 1), (3, -1)]),          # M3 = A11*(B12-B22)
        ([(3, 1)], [(2, 1), (0, -1)]),          # M4 = A22*(B21-B11)
        ([(0, 1), (1, 1)], [(3, 1)]),           # M5 = (A11+A12)*B22
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # M6 = (A21-A11)*(B11+B12)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # M7 = (A12-A22)*(B21+B22)
    ]
    
    # For each Strassen product, expand into schoolbook terms
    for a_coeffs, b_coeffs in strassen_products:
        for a_block, a_coeff in a_coeffs:
            for b_block, b_coeff in b_coeffs:
                coeff = a_coeff * b_coeff
                
                # Block a_block: 3x3 matrix
                # Local indices (i, j) -> global (row, col)
                # Block 0: rows 0-2, cols 0-2
                # Block 1: rows 0-2, cols 3-5
                # Block 2: rows 3-5, cols 0-2
                # Block 3: rows 3-5, cols 3-5
                
                a_row_offset = (a_block // 2) * 3
                a_col_offset = (a_block % 2) * 3
                b_row_offset = (b_block // 2) * 3
                b_col_offset = (b_block % 2) * 3
                
                # Schoolbook for 3x3: for each (i, j, k) in 0..2
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            # A global: (a_row_offset + i, a_col_offset + j)
                            # B global: (b_row_offset + j, b_col_offset + k)  -- mid index j
                            # C global: (a_row_offset + i, b_col_offset + k)
                            
                            a_row = a_row_offset + i
                            a_col = a_col_offset + j
                            b_row = b_row_offset + j
                            b_col = b_col_offset + k
                            c_row = a_row
                            c_col = b_col
                            
                            U[a_row * m + a_col, col] = coeff
                            V[b_row * p + b_col, col] = 1.0
                            W[c_col * n + c_row, col] = 1.0
                            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
