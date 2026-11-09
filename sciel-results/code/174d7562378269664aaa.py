"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Rank 56 decomposition using 4 blocks of rank-14 M_{2,4,2}
    # Each M_{2,4,2} is decomposed into 2 M_{2,2,2} Strassen products.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56

    # Strassen 2x2x2 products
    # (U_part, V_part, W_part)
    # Indices are (row, col) for 2x2 matrices
    strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_block(a_row_offset, b_row_offset, b_col_offset, start_col):
        """
        Adds a 2x2 * 2x2 Strassen decomposition.
        a_row_offset: row offset in A (0 or 2)
        b_row_offset: row offset in B (0 or 2)
        b_col_offset: col offset in B (0, 2, 4, 6)
        """
        c = start_col
        for uc, vc, wc in strassen:
            # U: n*m x rank. Index i*m + j
            for (ri, cj), coef in uc.items():
                gi = a_row_offset + ri
                gj = cj
                U[gi * m + gj][c] = coef
            
            # V: m*p x rank. Index j*p + k
            for (rj, ck), coef in vc.items():
                gj = b_row_offset + rj
                gk = b_col_offset + ck
                V[gj * p + gk][c] = coef
            
            # W: p*n x rank. Index k*n + i
            for (rk, ri2), coef in wc.items():
                gk = b_col_offset + rk
                gi = a_row_offset + ri2
                W[gk * n + gi][c] = coef
            
            c += 1
        return c

    col = 0
    # 4 blocks of 2x2 * 2x2
    # A is 2x4. Split into A_L (cols 0-1) and A_R (cols 2-3).
    # B is 4x8. Split rows into Top (0-1) and Bot (2-3).
    # Split cols into 4 blocks of width 2.
    
    for b_col in [0, 2, 4, 6]:
        # Block 1: A_L (rows 0-1, cols 0-1) * B_top (rows 0-1, cols b_col to b_col+1)
        # A_L is at a_row_offset=0, cols 0-1.
        # B_top is at b_row_offset=0, cols b_col...
        col = add_strassen_block(0, 0, b_col, col)
        
        # Block 2: A_R (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols b_col to b_col+1)
        # A_R is at a_row_offset=0? No, A_R uses cols 2-3 of A.
        # In add_strassen_block, a_row_offset is for rows of A.
        # We need to select cols 2-3 of A.
        # The function `add_strassen_block` currently assumes A block is at cols 0-1.
        # We need a col offset for A as well.
        
        # Let's refactor add_strassen_block to accept a_col_offset.

    # Redefine with a_col_offset
    def add_strassen_block_v2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, start_col):
        c = start_col
        for uc, vc, wc in strassen:
            for (ri, cj), coef in uc.items():
                gi = a_row_offset + ri
                gj = a_col_offset + cj
                U[gi * m + gj][c] = coef
            
            for (rj, ck), coef in vc.items():
                gj = b_row_offset + rj
                gk = b_col_offset + ck
                V[gj * p + gk][c] = coef
            
            for (rk, ri2), coef in wc.items():
                gk = b_col_offset + rk
                gi = a_row_offset + ri2
                W[gk * n + gi][c] = coef
            
            c += 1
        return c

    col = 0
    for b_col in [0, 2, 4, 6]:
        # A_L (cols 0-1) * B_top (rows 0-1)
        col = add_strassen_block_v2(0, 0, 0, b_col, col)
        # A_R (cols 2-3) * B_bot (rows 2-3)
        col = add_strassen_block_v2(0, 2, 2, b_col, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
