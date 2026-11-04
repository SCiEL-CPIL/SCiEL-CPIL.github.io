"""Matrix multiplication 4x4x6 with Strassen-based decomposition."""
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2x2: rank-7 decomposition
# A, B are 2x2 matrices indexed (i,j) with flat index i*2+j
# C = A * B, flat index i*2+j
# 7 rank-1 terms: (a_coeff, b_coeff, c_coeff) where each is a dict {flat_idx: coeff}
STRASSEN_2x2 = [
    # M1 = (A00+A11)*(B00+B11)
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: -1, 1: -1, 2: 1}),
    # M2 = (A10+A11)*B00
    ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),
    # M3 = A00*(B01-B11)
    ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: -1}),
    # M4 = A11*(B10-B00)
    ({3: 1}, {2: 1, 0: -1}, {2: 1, 3: 1}),
    # M5 = (A00+A01)*B11
    ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: 1}),
    # M6 = (A10-A00)*(B00+B01)
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: -1, 2: -1, 1: 1, 3: 1}),
    # M7 = (A01-A11)*(B10+B11)
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 1: 1, 2: -1, 3: 1}),
]

def build_4x4x2_strassen(n=4, m=4, p=2):
    """
    Build rank-28 decomposition for 4x4 x 4x2 -> 4x2.
    
    We decompose A (4x4) into four 2x2 blocks:
      A00: rows 0-1, cols 0-1  -> flat indices 0,1,4,5
      A01: rows 0-1, cols 2-3  -> flat indices 2,3,6,7
      A10: rows 2-3, cols 0-1  -> flat indices 8,9,12,13
      A11: rows 2-3, cols 2-3  -> flat indices 10,11,14,15
    
    B (4x2) into two 2x1 column-blocks... wait, B is 4x2.
    Let's think of B as having two 2x2 blocks side by side? No.
    
    Actually for 4x4 * 4x2:
    A is 4x4, B is 4x2, C is 4x2.
    
    Split rows of A into top-2 and bottom-2.
    Split cols of B... B is 4 rows x 2 cols.
    Split rows of B into top-2 and bottom-2.
    
    A_top (2x4) * B_top (2x2) -> contributes to C_top (2x2)
    A_top (2x4) * B_bot (2x2) -> contributes to C_bot (2x2)
    A_bot (2x4) * B_top (2x2) -> contributes to C_top (2x2)
    A_bot (2x4) * B_bot (2x2) -> contributes to C_bot (2x2)
    
    But A_top is 2x4, not 2x2. We need to handle this.
    
    Alternative: think of it as 2x2 block matrix multiplication.
    A = [[A00, A01], [A10, A11]] where each block is 2x2.
    B = [[B00], [B10]] where B00 is 2x2 (top 2 rows of B), B10 is 2x2 (bottom 2 rows of B).
    Wait, B is 4x2. So B_top is 2x2, B_bot is 2x2.
    
    C = A * B = [[A00*B_top + A01*B_bot], [A10*B_top + A11*B_bot]]
    C is 4x2. C_top = A00*B_top + A01*B_bot (2x2), C_bot = A10*B_top + A11*B_bot (2x2).
    
    So we need to compute:
    C_top = A00*B_top + A01*B_bot  (sum of two 2x2*2x2 products)
    C_bot = A10*B_top + A11*B_bot  (sum of two 2x2*2x2 products)
    
    Each 2x2*2x2 product has rank 7 via Strassen.
    Total: 4 products * rank 7 = rank 28.
    
    Let's map indices carefully.
    
    A flat index: i*4+j for i in 0..3, j in 0..3
    A00: i in {0,1}, j in {0,1} -> flat: 0,1,4,5
    A01: i in {0,1}, j in {2,3} -> flat: 2,3,6,7
    A10: i in {2,3}, j in {0,1} -> flat: 8,9,12,13
    A11: i in {2,3}, j in {2,3} -> flat: 10,11,14,15
    
    B flat index: i*2+j for i in 0..3, j in 0..1
    B_top: i in {0,1} -> flat: 0,1,2,3
    B_bot: i in {2,3} -> flat: 4,5,6,7
    
    C flat index: i*2+j for i in 0..3, j in 0..1
    C_top: i in {0,1} -> flat: 0,1,2,3
    C_bot: i in {2,3} -> flat: 4,5,6,7
    
    For each Strassen term in a 2x2*2x2 product, we need to map the 2x2 local indices to global flat indices.
    
    Let's define helper: for a 2x2 block of A with base offset (row_offset, col_offset) in the 4x4 grid,
    local (i,j) maps to global flat index (row_offset+i)*4 + (col_offset+j).
    
    Similarly for B: block with row_offset in 4-row grid, local (i,j) maps to (row_offset+i)*2 + j.
    For C: block with row_offset, local (i,j) maps to (row_offset+i)*2 + j.
    """
    U = {}
    V = {}
    W = {}
    
    # For each of the 4 sub-problems, apply Strassen
    # Sub-problem (a_block, b_block, c_block)
    # a_block: (row_offset, col_offset) in A's 2x2 block grid
    # b_block: row_offset in B's row grid (0 or 2)
    # c_block: row_offset in C's row grid (0 or 2)
    
    sub_problems = [
        ((0, 0), 0, 0),  # A00 * B_top -> C_top
        ((0, 1), 2, 0),  # A01 * B_bot -> C_top
        ((1, 0), 0, 2),  # A10 * B_top -> C_bot
        ((1, 1), 2, 2),  # A11 * B_bot -> C_bot
    ]
    
    for (a_row, a_col), b_row, c_row in sub_problems:
        for (a_coeff, b_coeff, c_coeff) in STRASSEN_2x2:
            u_new = {}
            v_new = {}
            w_new = {}
            for li, lc in a_coeff.items():
                gi = (a_row + li // 2) * 4 + (a_col + li % 2)
                u_new[gi] = u_new.get(gi, 0) + lc
            for li, lc in b_coeff.items():
                gi = (b_row + li // 2) * 2 + (li % 2)
                v_new[gi] = v_new.get(gi, 0) + lc
            for li, lc in c_coeff.items():
                gi = (c_row + li // 2) * 2 + (li % 2)
                w_new[gi] = w_new.get(gi, 0) + lc
            U[(a_row, a_col, b_row, c_row, id((a_coeff, b_coeff, c_coeff)))] = u_new
            V[(a_row, a_col, b_row, c_row, id((a_coeff, b_coeff, c_coeff)))] = v_new
            W[(a_row, a_col, b_row, c_row, id((a_coeff, b_coeff, c_coeff)))] = w_new
    
    # Convert to list of dicts
    keys = list(U.keys())
    U_list = [U[k] for k in keys]
    V_list = [V[k] for k in keys]
    W_list = [W[k] for k in keys]
    
    return U_list, V_list, W_list


def build_4x4x4_strassen(n=4, m=4, p=4):
    """
    Build rank-49 decomposition for 4x4 x 4x4 -> 4x4.
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = A*B, 4x4, split into four 2x2 blocks:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    8 sub-products * rank 7 = rank 56.
    
    But we can do better with full Strassen on 2x2 block matrix.
    The 2x2 block Strassen gives rank 7 for the block-level, each block product is 2x2*2x2.
    So total rank = 7 * 7 = 49.
    
    Block-level Strassen:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M2 + M6
    C01 = M3 + M5
    C10 = M2 + M7
    C11 = M1 - M3 + M5 - M7
    
    Each M is a 2x2*2x2 product. We apply Strassen to each M.
    Total: 7 * 7 = 49 rank-1 terms.
    """
    U = {}
    V = {}
    W = {}
    
    # Block offsets for A, B, C in their respective grids
    # A: 4x4, blocks at (row, col) in {(0,0),(0,1),(1,0),(1,1)}
    # B: 4x4, same
    # C: 4x4, same
    
    # For each block-level Strassen term, we have coefficients for A blocks, B blocks, C blocks
    block_strassen = [
        # (a_block_coeffs, b_block_coeffs, c_block_coeffs)
        # a_block_coeffs: dict {block_id: coeff}, block_id in {(0,0),(0,1),(1,0),(1,1)}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): -1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(1,0): 1, (1,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,1): 1, (1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(0,0): -1, (1,0): -1, (0,1): 1, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): -1, (0,1): 1, (1,0): -1, (1,1): 1}),
    ]
    
    term_id = 0
    for a_bc, b_bc, c_bc in block_strassen:
        # For each pair of A block and B block involved, create a 2x2*2x2 Strassen
        # Actually, we need to combine: for each (a_block, b_block) pair,
        # the coefficient is a_bc[a_block] * b_bc[b_block]
        # Then apply Strassen to the 2x2*2x2 product.
        
        for a_block, a_coeff in a_bc.items():
            for b_block, b_coeff in b_bc.items():
                combined_coeff = a_coeff * b_coeff
                if combined_coeff == 0:
                    continue
                
                # Now for this 2x2*2x2 product, apply Strassen
                for (sa_coeff, sb_coeff, sc_coeff) in STRASSEN_2x2:
                    u_new = {}
                    v_new = {}
                    w_new = {}
                    for li, lc in sa_coeff.items():
                        gi = (a_block[0] + li // 2) * 4 + (a_block[1] + li % 2)
                        u_new[gi] = u_new.get(gi, 0) + combined_coeff * lc
                    for li, lc in sb_coeff.items():
                        gi = (b_block[0] + li // 2) * 4 + (b_block[1] + li % 2)
                        v_new[gi] = v_new.get(gi, 0) + combined_coeff * lc
                    for li, lc in sc_coeff.items():
                        for c_block, c_coeff in c_bc.items():
                            gi = (c_block[0] + li // 2) * 4 + (c_block[1] + li % 2)
                            w_new[gi] = w_new.get(gi, 0) + combined_coeff * c_coeff * lc
                    
                    U[term_id] = u_new
                    V[term_id] = v_new
                    W[term_id] = w_new
                    term_id += 1
    
    keys = sorted(U.keys())
    U_list = [U[k] for k in keys]
    V_list = [V[k] for k in keys]
    W_list = [W[k] for k in keys]
    
    return U_list, V_list, W_list


def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: split 4x4x6 into 4x4x4 + 4x4x2
    # Rank 4x4x4 via Strassen: 49
    # Rank 4x4x2 via Strassen: 28
    # Total: 77
    
    # Build 4x4x4 part
    U44, V44, W44 = build_4x4x4_strassen()
    
    # Build 4x4x2 part
    U42, V42, W42 = build_4x4x2_strassen()
    
    # Combine
    # For 4x4x4: U is 16x49, V is 16x49, W is 16x49
    # For 4x4x2: U is 16x28, V is 8x28, W is 8x28
    # Combined: U is 16x77, V is 16x77 (pad V42 to 16 rows), W is 16x77 (pad W42 to 16 rows)
    
    rank_total = len(U44) + len(U42)
    
    U_full = [[0] * rank_total for _ in range(16)]
    V_full = [[0] * rank_total for _ in range(16)]
    W_full = [[0] * rank_total for _ in range(16)]
    
    # Fill 4x4x4 part
    for c, (u_dict, v_dict, w_dict) in enumerate(zip(U44, V44, W44)):
        for idx, coeff in u_dict.items():
            U_full[idx][c] = coeff
        for idx, coeff in v_dict.items():
            V_full[idx][c] = coeff
        for idx, coeff in w_dict.items():
            W_full[idx][c] = coeff
    
    # Fill 4x4x2 part
    for c, (u_dict, v_dict, w_dict) in enumerate(zip(U42, V42, W42)):
        col = len(U44) + c
        for idx, coeff in u_dict.items():
            U_full[idx][col] = coeff
        for idx, coeff in v_dict.items():
            # B is 4x2, flat index i*2+j, i in 0..3, j in 0..1 -> 8 entries
            # But we need 16 entries (4x4). The 4x2 part only uses 8 of them.
            # The other 8 entries (for cols 2,3 of B in a 4x4) are zero.
            V_full[idx][col] = coeff
        for idx, coeff in w_dict.items():
            # C is 4x2, flat index i*2+j -> 8 entries
            # Need 16 entries (4x4).
            W_full[idx][col] = coeff
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        pass  # Already done
    
    return {"U": U_full, "V": V_full, "W": W_full}
