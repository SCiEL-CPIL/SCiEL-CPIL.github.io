import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Rank 28 using Strassen for 2x2 * 2x2 blocks
    num_blocks = p // 2  # 4 blocks
    rank_per_block = 7   # Strassen rank
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for A (2x2) and M (2x2)
    # A indices: 00, 01, 10, 11
    # M indices: 00, 01, 10, 11
    # U indices (A): i*m+j
    # V indices (M): j*p+k (global)
    # W indices (C): k*n+i (global)
    
    # Define the 7 products P1..P7
    # Each is (coeffs_A, coeffs_M)
    # coeffs_A: dict or list of (idx, val)
    # coeffs_M: dict or list of (idx, val)
    
    # P1 = (A00 + A11) * (M00 + M11)
    # P2 = (A10 + A11) * M00
    # P3 = A00 * (M01 - M11)
    # P4 = A11 * (M10 - M00)
    # P5 = (A00 + A01) * M11
    # P6 = (A10 - A00) * (M00 + M01)
    # P7 = (A01 - A11) * (M10 + M11)
    
    # A indices:
    # A00: 0*2+0 = 0
    # A01: 0*2+1 = 1
    # A10: 1*2+0 = 2
    # A11: 1*2+1 = 3
    
    # M local indices:
    # M00: (0,0)
    # M01: (0,1)
    # M10: (1,0)
    # M11: (1,1)
    
    strassen_products = [
        # (list of (A_idx, coeff), list of (M_row, M_col, coeff))
        ([ (0, 1), (3, 1) ], [ (0, 0, 1), (1, 1, 1) ]), # P1
        ([ (2, 1), (3, 1) ], [ (0, 0, 1) ]),             # P2
        ([ (0, 1) ], [ (0, 1, 1), (1, 1, -1) ]),         # P3
        ([ (3, 1) ], [ (1, 0, 1), (0, 0, -1) ]),         # P4
        ([ (0, 1), (1, 1) ], [ (1, 1, 1) ]),             # P5
        ([ (2, 1), (0, -1) ], [ (0, 0, 1), (0, 1, 1) ]), # P6
        ([ (1, 1), (3, -1) ], [ (1, 0, 1), (1, 1, 1) ])  # P7
    ]
    
    # W contributions for each Pk to C00, C01, C10, C11
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    w_contribs = [
        # P1
        [ (0, 0, 1), (1, 1, 1) ], # C00: +1, C11: +1
        # P2
        [ (1, 0, 1), (1, 1, -1) ], # C10: +1, C11: -1
        # P3
        [ (0, 1, 1), (1, 1, 1) ], # C01: +1, C11: +1
        # P4
        [ (0, 0, 1), (1, 0, 1) ], # C00: +1, C10: +1
        # P5
        [ (0, 0, -1), (0, 1, 1) ], # C00: -1, C01: +1
        # P6
        [ (1, 1, 1) ], # C11: +1
        # P7
        [ (0, 0, 1) ] # C00: +1
    ]
    
    col = 0
    for t in range(num_blocks):
        for k_idx in range(rank_per_block):
            r = col
            a_coeffs, m_coeffs = strassen_products[k_idx]
            
            # Fill U
            for a_idx, coeff in a_coeffs:
                U[a_idx][r] = coeff
                
            # Fill V
            for m_row, m_col, coeff in m_coeffs:
                # Global B index: row = m_row, col = 2*t + m_col
                b_row = m_row
                b_col = 2 * t + m_col
                v_idx = b_row * p + b_col
                V[v_idx][r] = coeff
                
            # Fill W
            for c_row, c_col, coeff in w_contribs[k_idx]:
                # Global C index: row = c_row, col = 2*t + c_col
                c_row_global = c_row
                c_col_global = 2 * t + c_col
                # W index: k*n + i -> c_col_global * n + c_row_global
                w_idx = c_col_global * n + c_row_global
                W[w_idx][r] = coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
