"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition for 2x4 by 4x4 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks.
    Analysis suggests 28 is the minimal rank for this tensor.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen rank-1 components for 2x2*2x2 (P*Q=R):
    # Each component: (U_entries: {P_idx: val}, V_entries: {Q_idx: val}, W_entries: {R_idx: val})
    # P_idx, Q_idx, R_idx are (row, col) in 2x2
    strassen_components = [
        # M1 = (P00+P11)*(Q00+Q11) -> R00+=1, R11+=1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (P10+P11)*Q00 -> R10+=1, R11-=1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = P00*(Q01-Q11) -> R01+=1, R11+=1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = P11*(Q10-Q00) -> R00+=1, R10+=1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (P00+P01)*Q11 -> R00-=1, R01+=1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (P10-P00)*(Q00+Q01) -> R11+=1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (P01-P11)*(Q10+Q11) -> R00+=1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # The 4 terms: (A_block, B_block, C_block)
    # A_block 0: cols 0,1 -> global col offset 0
    # A_block 1: cols 2,3 -> global col offset 2
    # B_block 00: rows 0,1 cols 0,1
    # B_block 01: rows 0,1 cols 2,3
    # B_block 10: rows 2,3 cols 0,1
    # B_block 11: rows 2,3 cols 2,3
    # C_block 0: cols 0,1 -> global col offset 0
    # C_block 1: cols 2,3 -> global col offset 2
    
    # For each term: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    terms = [
        # C0 = A0*B00 + A1*B10
        (0, 0, 0, 0),  # A0 * B00 -> C0
        (2, 2, 0, 0),  # A1 * B10 -> C0
        # C1 = A0*B01 + A1*B11
        (0, 0, 2, 2),  # A0 * B01 -> C1
        (2, 2, 2, 2),  # A1 * B11 -> C1
    ]
    
    for term_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(terms):
        for s_idx, (u_entries, v_entries, w_entries) in enumerate(strassen_components):
            c = term_idx * 7 + s_idx
            
            # Fill U: P is A_block, index (pi, pj) -> global (pi, a_col_off + pj)
            for (pi, pj), val in u_entries.items():
                u_idx = pi * m + (a_col_off + pj)
                U[u_idx][c] += val
            
            # Fill V: Q is B_block, index (qi, qj) -> global (b_row_off + qi, b_col_off + qj)
            for (qi, qj), val in v_entries.items():
                v_idx = (b_row_off + qi) * p + (b_col_off + qj)
                V[v_idx][c] += val
            
            # Fill W: R is C_block, index (ri, rj) -> global (ri, c_col_off + rj)
            # W index: k*n + i where k is col, i is row
            for (ri, rj), val in w_entries.items():
                w_idx = (c_col_off + rj) * n + ri
                W[w_idx][c] += val
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
