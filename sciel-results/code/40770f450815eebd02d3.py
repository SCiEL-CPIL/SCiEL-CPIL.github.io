"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen on 2x2 blocks
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products definition
    # M indices: 00, 01, 10, 11 -> u_idx
    # N indices: 00, 01, 10, 11 -> v_idx
    # P indices: 00, 01, 10, 11 -> w_idx
    # Each product k has coefficients for M, N, P
    
    # We define the 7 Strassen products as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {m_idx: coeff}
    # v_coeffs: dict {n_idx: coeff}
    # w_coeffs: dict {p_idx: coeff}
    
    strassen_products = [
        # p1 = (m00 + m11)(n00 + n11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # p2 = (m10 + m11)n00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # p3 = m00(n01 - n11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # p4 = m11(n10 - n00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # p5 = (m00 + m01)n11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # p6 = (m10 - m00)(n00 + n01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # p7 = (m01 - m11)(n10 + n11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Map local 2x2 indices to global indices
    # M (A) is 2x4. Local 2x2 block (row_off, col_off)
    # N (B) is 4x6. Local 2x2 block (row_off, col_off)
    # P (C) is 2x6. Local 2x2 block (row_off, col_off)
    
    # We have 3 blocks of B (col blocks of size 2)
    # For each block, we have 2 sub-multiplications (A1*B1 and A2*B2)
    
    col = 0
    for b in range(3): # Block index in B (0, 1, 2)
        b_k_start = b * 2
        
        # Sub-mult 1: A1 * B1
        # A1: rows 0-1, cols 0-1
        # B1: rows 0-1, cols b_k_start to b_k_start+1
        # Output: rows 0-1, cols b_k_start to b_k_start+1
        
        a_row_off = 0
        a_col_off = 0
        b_row_off = 0
        b_col_off = b_k_start
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_products:
            for m_idx, coeff in u_coeffs.items():
                # m_idx 0:00, 1:01, 2:10, 3:11
                r, c = divmod(m_idx, 2)
                u_row = (a_row_off + r) * m + (a_col_off + c)
                U[u_row][col] += coeff
                
            for n_idx, coeff in v_coeffs.items():
                r, c = divmod(n_idx, 2)
                v_row = (b_row_off + r) * p + (b_col_off + c)
                V[v_row][col] += coeff
                
            for p_idx, coeff in w_coeffs.items():
                r, c = divmod(p_idx, 2)
                # W index: k*n + i. Here i is row (r), k is col (c)
                w_row = (b_col_off + c) * n + (a_row_off + r)
                W[w_row][col] += coeff
                
            col += 1

        # Sub-mult 2: A2 * B2
        # A2: rows 0-1, cols 2-3
        # B2: rows 2-3, cols b_k_start to b_k_start+1
        # Output: rows 0-1, cols b_k_start to b_k_start+1
        
        a_row_off = 0
        a_col_off = 2
        b_row_off = 2
        b_col_off = b_k_start
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_products:
            for m_idx, coeff in u_coeffs.items():
                r, c = divmod(m_idx, 2)
                u_row = (a_row_off + r) * m + (a_col_off + c)
                U[u_row][col] += coeff
                
            for n_idx, coeff in v_coeffs.items():
                r, c = divmod(n_idx, 2)
                v_row = (b_row_off + r) * p + (b_col_off + c)
                V[v_row][col] += coeff
                
            for p_idx, coeff in w_coeffs.items():
                r, c = divmod(p_idx, 2)
                w_row = (b_col_off + c) * n + (a_row_off + r)
                W[w_row][col] += coeff
                
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
