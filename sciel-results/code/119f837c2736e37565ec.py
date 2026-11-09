"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strategy: Decompose 2x4 * 4x8 using block Strassen.
    # Split A into two 2x2 blocks: A = [A_L | A_R]
    # Split B into four 4x2 blocks: B = [B_0 | B_1 | B_2 | B_3]
    # C[:, k:k+2] = A_L @ B_k[0:2,:] + A_R @ B_k[2:4,:]
    # Each term is 2x2 * 2x2 = 2x2, using Strassen rank 7.
    # Total: 4 blocks * 2 sub-blocks * 7 = 56
    #
    # To reduce rank, we exploit shared A structure:
    # Precompute: S1 = A_L + A_R, S2 = A_L - A_R, D1 = A_L, D2 = A_R
    # For each B_k, split into B_k_top (2x2) and B_k_bot (2x2)
    # C_k = A_L @ B_k_top + A_R @ B_k_bot
    #
    # Using Strassen on each pair gives 7 per pair, 4 pairs = 28 rank terms.
    # But we need 2 independent 2x2 products per B_k block, so 4*2*7=56.
    #
    # Alternative: treat each 2x4 @ 4x2 as a single unit.
    # 2x4 @ 4x2 has rank at most 2*4*2=16, but can be done in fewer.
    # Actually, 2x4 @ 4x2 = (2x2 @ 2x2) + (2x2 @ 2x2) structure.
    #
    # Let's use a rank-7 decomposition for each 2x2x2, giving 8*7=56.
    # This is the current approach. Let's verify it's correct.

    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56

    # Strassen 2x2x2 products
    strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_block(a_col, b_row, b_col, start_col):
        c = start_col
        for uc, vc, wc in strassen:
            for (ri, cj), coef in uc.items():
                gi = ri
                gj = a_col + cj
                U[gi * m + gj][c] = coef
            for (rj, ck), coef in vc.items():
                gj = b_row + rj
                gk = b_col + ck
                V[gj * p + gk][c] = coef
            for (rk, ri2), coef in wc.items():
                gk = b_col + rk
                gi = ri2
                W[gk * n + gi][c] = coef
            c += 1
        return c

    col = 0
    for b_col in [0, 2, 4, 6]:
        col = add_block(0, 0, b_col, col)
        col = add_block(2, 2, b_col, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
