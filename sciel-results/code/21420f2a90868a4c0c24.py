"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Rank 36 solution:
    # Split B (4x6) into two 4x3 blocks B1, B2.
    # For each 4x3 block, use rank 18 decomposition of 2x4x3.
    # 2x4x3 decomposition:
    # Split A (2x4) into A1, A2 (2x2).
    # Split B_block (4x3) into B_top, B_bot (2x3).
    # C = A1*B_top + A2*B_bot.
    # Each 2x2x3 multiplication uses rank 9:
    #   Split B_2x3 into B_2x2 and b_2x1.
    #   A*B_2x2 uses Strassen (rank 7).
    #   A*b_2x1 uses direct (rank 2).
    # Total rank = 2 * (2 * 9) = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 36)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 36)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 36)
    
    # Strassen coefficients for 2x2x2
    # U: A (2x2), V: B (2x2), W: C (2x2)
    # P1: A00+A11, P2: A10+A11, P3: A00, P4: A11, P5: A00+A01, P6: A10-A00, P7: A01-A11
    U_strassen = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_strassen = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_strassen = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]
    
    col = 0
    # Two 4x3 blocks of B
    for b_block_idx in range(2):
        # 2x4x3 decomposition for this block
        # A is split into A1 (cols 0-1), A2 (cols 2-3)
        for a_block_idx in range(2):
            # B_block is split into B_top (rows 0-1), B_bot (rows 2-3)
            for b_row_block_idx in range(2):
                # 2x2x3 multiplication: A_sub (2x2) * B_sub (2x3)
                # B_sub is split into B_2x2 (cols 0-1) and b_2x1 (col 2)
                # Part 1: A_sub * B_2x2 (Strassen, rank 7)
                for s in range(7):
                    # U: A_sub indices (i, j) where j is local col in A
                    # A_sub is A[a_block_idx*2 : a_block_idx*2+2, :]? No.
                    # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
                    # So A_sub is A[:, a_block_idx*2 : a_block_idx*2+2].
                    # Local index (i_loc, j_loc) -> global i=i_loc, j=a_block_idx*2+j_loc.
                    for (i_loc, j_loc), coeff in U_strassen[s].items():
                        i = i_loc
                        j = a_block_idx * 2 + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += coeff
                    
                    # V: B_sub indices.
                    # B_block is 4x3. B_top is rows 0,1. B_bot is rows 2,3.
                    # B_sub is B_block[b_row_block_idx*2 : b_row_block_idx*2+2, 0:2] (2x2 part).
                    # Local index (j_loc, k_loc) -> global j=b_row_block_idx*2+j_loc, k=k_loc.
                    # Note: k is local to the 4x3 block. Global k = b_block_idx*3 + k_loc.
                    for (j_loc, k_loc), coeff in V_strassen[s].items():
                        j = b_row_block_idx * 2 + j_loc
                        k = b_block_idx * 3 + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    
                    # W: Output C_sub (2x2).
                    # C is 2x6.
                    # C_sub corresponds to cols k in 0,1 of the 4x3 block.
                    # Global k = b_block_idx*3 + k_loc.
                    # i is i_loc.
                    for (i_loc, k_loc), coeff in W_strassen[s].items():
                        i = i_loc
                        k = b_block_idx * 3 + k_loc
                        w_idx = k * n + i
                        W[w_idx][col] += coeff
                    col += 1
                
                # Part 2: A_sub * b_2x1 (Direct, rank 2)
                # b_2x1 is B_block[b_row_block_idx*2 : b_row_block_idx*2+2, 2]
                # C_sub_col is C[:, b_block_idx*3 + 2]
                for i_loc in range(2):
                    for j_loc in range(2):
                        # Term: A[i_loc, j_loc] * B[j, 2]
                        # U
                        i = i_loc
                        j = a_block_idx * 2 + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += 1
                        # V
                        j_global = b_row_block_idx * 2 + j_loc
                        k_global = b_block_idx * 3 + 2
                        v_idx = j_global * p + k_global
                        V[v_idx][col] += 1
                        # W
                        k = b_block_idx * 3 + 2
                        w_idx = k * n + i
                        W[w_idx][col] += 1
                        col += 1
    
    if time.time() - start_time > 130:
        raise TimeoutError()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
