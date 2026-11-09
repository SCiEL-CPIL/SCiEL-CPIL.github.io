"""Matrix multiplication 2,4,8 tensor decomposition"""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is n x p
    
    # Rank 56
    # Structure: 4 blocks along p (k), each block is a 2x4x2 multiplication.
    # Each 2x4x2 multiplication is decomposed into two 2x2x2 Strassen multiplications.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: (a00+a11)(b00+b11) -> C00 +=, C11 +=
    # u: (0,0):1, (1,1):1
    # v: (0,0):1, (1,1):1
    # w: (0,0):1, (1,1):1
    # M2: (a10+a11)b00 -> C10 +=, C11 -=
    # u: (1,0):1, (1,1):1
    # v: (0,0):1
    # w: (0,1):1, (1,1):-1
    # M3: a00(b01-b11) -> C01 +=, C11 +=
    # u: (0,0):1
    # v: (0,1):1, (1,1):-1
    # w: (1,0):1, (1,1):1
    # M4: a11(b10-b00) -> C00 +=, C10 +=
    # u: (1,1):1
    # v: (1,0):1, (0,0):-1
    # w: (0,0):1, (0,1):1
    # M5: (a00+a01)b11 -> C00 -=, C01 +=
    # u: (0,0):1, (0,1):1
    # v: (1,1):1
    # w: (0,0):-1, (1,0):1
    # M6: (a10-a00)(b00+b01) -> C11 +=
    # u: (1,0):1, (0,0):-1
    # v: (0,0):1, (0,1):1
    # w: (1,1):1
    # M7: (a01-a11)(b10+b11) -> C00 +=
    # u: (0,1):1, (1,1):-1
    # v: (1,0):1, (1,1):1
    # w: (0,0):1

    strassen_u = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    strassen_v = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    strassen_w = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    # Helper to add a Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset, base_col):
        # i_offset: 0 or 1 (row offset in 2x2 block of A)
        # j_offset: 0 or 1 (col offset in 2x2 block of A, and row in B)
        # k_offset: 0 or 1 (col offset in 2x2 block of B, and row in C)
        for r in range(7):
            for (i_loc, j_loc), val in strassen_u[r].items():
                i_glob = i_offset + i_loc
                j_glob = j_offset + j_loc
                U[i_glob * m + j_glob][base_col + r] += val
                
            for (j_loc, k_loc), val in strassen_v[r].items():
                j_glob = j_offset + j_loc
                k_glob = k_offset + k_loc
                V[j_glob * p + k_glob][base_col + r] += val
                
            for (k_loc, i_loc), val in strassen_w[r].items():
                k_glob = k_offset + k_loc
                i_glob = i_offset + i_loc
                W[k_glob * n + i_glob][base_col + r] += val

    # Main loop over k blocks (4 blocks of size 2)
    for t in range(4):
        k_base = 2 * t
        
        # For this k block, we compute C[:, 2t:2t+2] = A * B[:, 2t:2t+2]
        # A is 2x4. B_block is 4x2.
        # Split A into A_L (cols 0,1) and A_R (cols 2,3).
        # Split B_block into B_top (rows 0,1) and B_bot (rows 2,3).
        # C = A_L * B_top + A_R * B_bot
        
        # Part 1: A_L * B_top
        # A_L is rows 0,1 of A, cols 0,1. (i_offset=0, j_offset=0)
        # B_top is rows 0,1 of B_block, cols 0,1. (j_offset=0, k_offset=0 relative to block)
        # Global j for B_top rows: 0,1. Global k for B_top cols: k_base, k_base+1.
        add_strassen_block(0, 0, k_base, col)
        col += 7
        
        # Part 2: A_R * B_bot
        # A_R is rows 0,1 of A, cols 2,3. (i_offset=0, j_offset=2)
        # B_bot is rows 2,3 of B_block, cols 0,1. (j_offset=2, k_offset=0 relative to block)
        # Global j for B_bot rows: 2,3. Global k for B_bot cols: k_base, k_base+1.
        add_strassen_block(0, 2, k_base, col)
        col += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
