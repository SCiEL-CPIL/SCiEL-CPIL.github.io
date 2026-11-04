DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition for 2x4x8 matrix multiplication
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We use a structured decomposition:
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x8) into B1 (2x8) and B2 (2x8)
    # Split B1 into B1a (2x4) and B1b (2x4)
    # Split B2 into B2a (2x4) and B2b (2x4)
    # 
    # Result C = A1*B1 + A2*B2
    #         = A1*(B1a+B1b) + A2*(B2a+B2b)
    #         = A1*B1a + A1*B1b + A2*B2a + A2*B2b
    #
    # Each term is 2x2 times 2x4 = 2x4
    # Use Strassen for each 2x2x2 block within the 2x4
    # 2x4 = 2*(2x2), so each 2x2x2x4 needs 7*2=14? No...
    # Actually 2x2 * 2x4: we can split the 2x4 into two 2x2 blocks
    # So each 2x2 * 2x4 = (2x2 * 2x2) + (2x2 * 2x2)
    # Each 2x2*2x2 uses Strassen rank 7
    # So each 2x2*2x4 uses rank 14
    # Total: 4 * 14 = 56. That's the same as before.
    #
    # Let me try a different approach: use rank 48 directly.
    # 48 = 6 * 8 or 4 * 12 or 3 * 16
    #
    # Alternative: Use the fact that 2*4*8 = 64 entries
    # We need at least ceil(64/something) 
    #
    # Let me try a direct construction with rank 48.
    # I'll use a pattern based on splitting differently.
    #
    # Actually, let me use a known lower bound approach.
    # The tensor rank of matrix multiplication <m,n,p> 
    # For 2x4x8, the standard is 2*4*8=64
    # Strassen-based gives 56
    # 
    # Let me try to implement a rank 48 decomposition using a different splitting.
    # Split A into 2x2 blocks: A = [A1 A2] where A1, A2 are 2x2
    # Split B into 4x8: keep as is
    # C = A1*B + A2*B where B is 4x8
    # Each A_i * B is 2x4 * 4x8 = 2x8
    # 
    # For 2x4 * 4x8: split B into B_top (2x8) and B_bot (2x8)
    # A_i * B = A_i_top * B_top + A_i_bot * B_bot
    # where A_i_top is 2x2 (top half of A_i) and A_i_bot is 2x2 (bottom half)
    # 
    # So we have 4 terms of 2x2 * 2x8
    # Each 2x2 * 2x8: split 2x8 into 4 blocks of 2x2
    # Each 2x2*2x2 uses Strassen rank 7
    # So each 2x2*2x8 uses rank 28
    # Total: 4 * 28 = 112. Worse.
    #
    # Let me just use the rank 56 Strassen approach but verify it's correct.
    # Actually, let me try rank 48 with a different structure.
    #
    # I'll use a direct enumeration approach for small cases.
    # For now, I'll output rank 56 as it's the best I can reliably construct.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def add_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_off, lj + j_off
                U[gi * 4 + gj][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_off, lk + k_off
                V[gj * 8 + gk][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_off, li + i_off
                W[gk * 2 + gi][cr] += val
    
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
