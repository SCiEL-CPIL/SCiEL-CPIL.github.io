import numpy as np

def construct():
    n, m, p = 4, 4, 7
    # Use Strassen 2x2x2 (rank 7) where each scalar multiply is a 2x2 x 2x7 schoolbook (rank 14)
    # Total rank = 7 * 14 = 98
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 28 x 98
    W = [[0] * rank for _ in range(p * n)]  # 28 x 98

    # Strassen 2x2x2 coefficients
    # For term t, U2[row][t], V2[row][t], W2[row][t]
    # Rows 0-3 correspond to A00,A01,A10,A11 / B00,B01,B10,B11 / C00,C01,C10,C11
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ]

    # Map 2x2 block indices to global indices
    # A: 4x4, blocks 2x2. A[i,j] where i,j in {0,1}
    # A block (bi,bj) has global row indices: bi*2 + r for r in {0,1}, col: bj*2 + c for c in {0,1}
    # A index = (bi*2 + r) * m + (bj*2 + c) = (bi*2+r)*4 + (bj*2+c)
    # B: 4x7, blocks 2x7. B block (bi,bj) where bi in {0,1}, bj in {0,1} for the 2x2 part
    # But B is 4x7, so we split columns into two groups of 2+2+2+1? 
    # Actually, let's split B into 2x2 blocks for the first 4 columns, and handle cols 4-6 separately
    # Better: treat B as 2x2 blocks where each block is 2 rows x (7/2) cols... 
    # 
    # Let me reconsider: The Strassen decomposition treats the 4x4 A as 2x2 blocks of 2x2 matrices.
    # For B (4x7), we need to split it into 2x2 blocks where each block is 2 rows x k columns.
    # Since 7 is odd, let's split B into blocks of width 2,2,2,1.
    # But Strassen only works with 2x2 block structure.
    #
    # Alternative: Apply Strassen to the 2x2 block structure of A and B where B is padded to 4x8,
    # then discard the 8th column. But that changes the problem.
    #
    # Let me use a different approach: 
    # View the 4x4 x 4x7 multiplication as 2x2 block multiplication where:
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] each 2x2 (first 3) and 2x1 (last)
    # 
    # For the first 3 column blocks (width 2 each), we can use Strassen.
    # For the last column block (width 1), we use schoolbook.
    #
    # Actually, let me just do: for each of the 4 column positions in B (each of width ~2),
    # apply the 2x2 Strassen to the 2x2 block rows of A and B.
    #
    # Simpler: Just use schoolbook for the full 4x4x7 but with Strassen optimization on the 2x2 block level.
    # 
    # Let me just implement it directly:
    # For each Strassen term t (0..6), and for each "inner" schoolbook term (i,j,k) where
    # i in {0,1} (row within 2x2 block), j in {0,1} (col within 2x2 block of A), k in {0..6} (col in B)
    # 
    # A index: (a_block_row * 2 + i) * 4 + (a_block_col * 2 + j)
    # B index: (b_block_row * 2 + i) * 7 + k  -- wait, this doesn't work cleanly
    #
    # Let me just do the straightforward Kronecker-like approach:
    # The 2x2x2 Strassen gives 7 terms. Each term multiplies a 2x2 "matrix" from A-side
    # with a 2x7 "matrix" from B-side to produce a 2x7 "matrix" in C-side.
    # The 2x2 x 2x7 multiplication has rank 2*2*7 = 28 by schoolbook, but we can do it
    # as 2 (rows of A-block) * 2 (cols of A-block) * 7 (cols of B) = 28 rank-1 terms.
    # Total: 7 * 28 = 196. That's worse than schoolbook 112.
    #
    # I think the correct approach is:
    # 4x4 x 4x7: Split A into 2x2 blocks (4 blocks of 2x2). Split B into 2x2 blocks for cols 0-3,
    # and 2x1 for col 4-6... this is messy.
    #
    # Use Strassen for 4x4x4 (rank 49) + schoolbook for remaining 3 columns (rank 48) = 97
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen 4x4x4 for first 4 columns of B
    # A blocks: A00=[0,1,4,5], A01=[2,3,6,7], A10=[8,9,12,13], A11=[10,11,14,15]
    # B blocks: B00=[0,1,4,5], B01=[2,3,6,7], B10=[8,9,12,13], B11=[10,11,14,15]
    # C blocks: C00=[0,1,8,9], C01=[2,3,10,11], C10=[16,17,24,25], C11=[18,19,26,27]
    
    A_blocks = {0: [0,1,4,5], 1: [2,3,6,7], 2: [8,9,12,13], 3: [10,11,14,15]}
    B_blocks = {0: [0,1,4,5], 1: [2,3,6,7], 2: [8,9,12,13], 3: [10,11,14,15]}
    C_blocks = {0: [0,1,8,9], 1: [2,3,10,11], 2: [16,17,24,25], 3: [18,19,26,27]}
    
    # Strassen terms: (a_combo, b_combo, c_combos)
    # Each combo is a list of (block_idx, sign) pairs
    strassen_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),      # (A00+A11)(B00+B11) -> C00, C11
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),                    # (A10+A11)B00 -> C10
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),                   # A00(B01-B11) -> C01
        ([(3,1)], [(2,1),(0,-1)], [(2,1)]),                   # A11(B10-B00) -> C10
        ([(0,1),(1,1)], [(3,1)], [(0,1),(1,1)]),             # (A00+A01)B11 -> C00, C01
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),            # (A10-A00)(B00+B01) -> C11
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,1)]),      # (A01-A11)(B10+B11) -> C00, C11
    ]
    
    col = 0
    for a_combo, b_combo, c_combos in strassen_terms:
        # For each of the 4x4x4 schoolbook terms within this Strassen term
        # The Strassen term combines A blocks and B blocks, then multiplies
        # We need to expand: for each element in the combined A, each element in combined B
        a_indices = []
        for block_idx, sign in a_combo:
            for idx in A_blocks[block_idx]:
                a_indices.append((idx, sign))
        
        b_indices = []
        for block_idx, sign in b_combo:
            for idx in B_blocks[block_idx]:
                b_indices.append((idx, sign))
        
        # For each pair (a_idx, b_idx), we have a rank-1 term
        for a_idx, a_sign in a_indices:
            for b_idx, b_sign in b_indices:
                # C index: the product goes to all C blocks in c_combos
                # But wait, this isn't right. The Strassen term computes one product,
                # and that product contributes to multiple C blocks.
                # 
                # Actually, each Strassen term is: (sum of A blocks) * (sum of B blocks) = product
                # The product is a 4x4 matrix that gets added to certain C blocks.
                # 
                # Let me reconsider. The 2x2x2 Strassen operates on 2x2 blocks.
                # Each "element" in the 2x2 matrices is a 2x2 submatrix.
                # So the product of two "elements" is a 2x2 * 2x2 = 4x4... no, 2x2*2x2=2x2.
                # 
                # I think I need to be more careful. Let me just do the full expansion.
                # 
                # For now, let me just use schoolbook for everything to be safe.
                pass
        col += 4  # placeholder, not correct
    
    # Schoolbook: rank = n*m*p = 112
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
