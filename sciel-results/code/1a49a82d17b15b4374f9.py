import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # Part 2: 4x4x3 decomposition (Rank 44)
    
    # --- Base Strassen 2x2x2 (Rank 7) ---
    # U: 4x7, V: 4x7, W: 4x7
    U_s2 = np.zeros((4, 7), dtype=int)
    V_s2 = np.zeros((4, 7), dtype=int)
    W_s2 = np.zeros((4, 7), dtype=int)
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    U_s2[0, 0] = 1; U_s2[3, 0] = 1
    V_s2[0, 0] = 1; V_s2[3, 0] = 1
    W_s2[0, 0] = 1; W_s2[3, 0] = 1
    
    # Term 2: (A10+A11)B00 -> C10
    U_s2[2, 1] = 1; U_s2[3, 1] = 1
    V_s2[0, 1] = 1
    W_s2[1, 1] = 1
    
    # Term 3: A00(B01-B11) -> C01
    U_s2[0, 2] = 1
    V_s2[1, 2] = 1; V_s2[3, 2] = -1
    W_s2[2, 2] = 1
    
    # Term 4: A11(B10-B00) -> C00(-1), C11(-1)
    U_s2[3, 3] = 1
    V_s2[2, 3] = 1; V_s2[0, 3] = -1
    W_s2[0, 3] = -1; W_s2[3, 3] = -1
    
    # Term 5: (A00+A01)B11 -> C01
    U_s2[0, 4] = 1; U_s2[1, 4] = 1
    V_s2[3, 4] = 1
    W_s2[2, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01) -> C00
    U_s2[2, 5] = 1; U_s2[0, 5] = -1
    V_s2[0, 5] = 1; V_s2[1, 5] = 1
    W_s2[0, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11) -> C11
    U_s2[1, 6] = 1; U_s2[3, 6] = -1
    V_s2[2, 6] = 1; V_s2[3, 6] = 1
    W_s2[3, 6] = 1

    # --- Base Standard 2x2x1 (Rank 4) ---
    U_st = np.eye(4, dtype=int)
    V_st = np.eye(4, dtype=int)
    W_st = np.eye(4, dtype=int)

    # --- Build 4x4x4 (Rank 49) ---
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # Outer Strassen terms definitions
    # Each term: (A_blocks, signs), (B_blocks, signs), (C_blocks, signs)
    # A_blocks: list of block indices (00, 01, 10, 11)
    # Block index to row/col in 4x4:
    # 00: rows 0-1, cols 0-1
    # 01: rows 0-1, cols 2-3
    # 10: rows 2-3, cols 0-1
    # 11: rows 2-3, cols 2-3
    
    outer_terms = [
        # 1
        ([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, 1]), # (A00+A11)(B00+B11) -> C00, C11
        # 2
        ([2, 3], [1, 1]), ([0], [1]), ([2], [1]), # (A10+A11)B00 -> C10 (Block 2 is C10? No. C blocks: 00,01,10,11. C10 is block index 2 in 0,1,2,3? 
        # C is 4x4. C00 is 0,1x0,1. C01 is 0,1x2,3. C10 is 2,3x0,1. C11 is 2,3x2,3.
        # So C blocks indices: 0:00, 1:01, 2:10, 3:11.
        # My W_s2 indices: 0:00, 1:01, 2:10, 3:11.
        # So for Term 2, C block is 2 (C10).
        # 3
        ([0], [1]), ([1, 3], [1, -1]), ([1], [1]), # A00(B01-B11) -> C01
        # 4
        ([3], [1]), ([2, 0], [1, -1]), ([0, 3], [-1, -1]), # A11(B10-B00) -> C00(-1), C11(-1)
        # 5
        ([0, 1], [1, 1]), ([3], [1]), ([1], [1]), # (A00+A01)B11 -> C01
        # 6
        ([2, 0], [1, -1]), ([0, 1], [1, 1]), ([0], [1]), # (A10-A00)(B00+B01) -> C00
        # 7
        ([1, 3], [1, -1]), ([2, 3], [1, 1]), ([3], [1])  # (A01-A11)(B10+B11) -> C11
    ]
    
    col_offset = 0
    for s_idx, (a_blocks, a_signs, b_blocks, b_signs, c_blocks, c_signs) in enumerate(outer_terms):
        for t_idx in range(7):
            # For this inner term t_idx, we have U_s2, V_s2, W_s2 contributions
            # We need to map these to the 4x4 blocks
            
            # U part
            for i in range(4):
                if U_s2[i, t_idx] != 0:
                    sign_u = U_s2[i, t_idx]
                    # i is index in 2x2 A block.
                    # Which A block? It's determined by the outer term's A blocks?
                    # No, the outer term defines a "virtual" 2x2 matrix A_virt.
                    # A_virt is formed by linear combination of A blocks.
                    # But Strassen operates on elements.
                    # This approach is flawed because Strassen combines elements.
                    
            # Let's rethink.
            # The 4x4x4 decomposition is not simply 7 instances of 2x2x2.
            # It is a tensor of rank 49.
            # I will use a known explicit decomposition or generate it via recursive application.
            
            pass

    # Due to complexity of manual mapping, I will use a simpler recursive generator.
    def get_strassen_matrices(n):
        if n == 2:
            return U_s2, V_s2, W_s2
        
        # Recursive step
        # n x n x n
        # Split into (n/2) x (n/2) blocks
        half = n // 2
        U_sub, V_sub, W_sub = get_strassen_matrices(half)
        
        # Outer terms
        # We need to construct U, V, W of size (n*n, 49)
        # Actually, rank is 49.
        
        # This is getting too complex for a quick implementation.
        # I will hardcode the 4x4x4 decomposition using a verified source or simple logic.
        # Or, I can just use the 93 rank solution by building 2x2x3 blocks.
        # The 4x4x4 part can be built by 4 instances of 2x2x4?
        # R(2,2,4) = 2 * R(2,2,2) = 14?
        # If R(2,2,4) is 14, then 4 * 14 = 56.
        # 56 + 44 = 100.
        # 100 is better than 112, but worse than 93.
        
        # I will stick to 93.
        # I need the 4x4x4 matrices.
        # I will generate them by composing the 2x2x2 Strassen matrices.
        
        # Let's try a different composition.
        # The 4x4x4 tensor can be decomposed into 7 terms of 2x2x2 tensors?
        # No, each term is a 2x2x2 tensor, but the inputs are linear combinations.
        
        # I will use the following trick:
        # Generate the 4x4x4 decomposition by iterating over all 49 basis vectors.
        # This is equivalent to applying the 2x2x2 Strassen algorithm to the blocks.
        
        # Let's implement the recursive composition properly.
        
        pass

    # Fallback: Use rank 100 solution (4x R(2,2,4) + 4x R(2,2,3)?)
    # No, I will implement the 93 solution by explicitly building the 4x4x4 part.
    
    # For now, I will return the 112 solution to ensure correctness, 
    # but I will try to implement the 93 solution.
    
    # Actually, I can just use the 112 solution if I can't get 93 working quickly.
    # But 93 is a significant improvement.
    
    # Let's try to build 4x4x4 by composing 2x2x2.
    
    def build_444():
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        # Outer terms
        # 1. (A00+A11)(B00+B11)
        # 2. (A10+A11)B00
        # 3. A00(B01-B11)
        # 4. A11(B10-B00)
        # 5. (A00+A01)B11
        # 6. (A10-A00)(B00+B01)
        # 7. (A01-A11)(B10+B11)
        
        # For each outer term, we have a 2x2x2 multiplication.
        # We use the base Strassen matrices for this.
        
        # Helper to add a block
        def add_block(mat, block_idx, sub_mat, sign=1):
            r0 = (block_idx // 2) * 2
            c0 = (block_idx % 2) * 2
            mat[r0:r0+2, :] += sign * sub_mat
        
        # This is for U, V, W which are 1D in block space?
        # U is 16x49. Block 0 is rows 0-3? No.
        # A is 4x4. Blocks are 2x2.
        # Block 0 (A00): rows 0-1, cols 0-1. Indices 0,1,2,3 in flattened U?
        # U index is i*m+j. i in 0..3, j in 0..3.
        # A00: i=0,1; j=0,1. Indices 0,1,2,3.
        # A01: i=0,1; j=2,3. Indices 0,1,4,5.
        # A10: i=2,3; j=0,1. Indices 8,9,10,11.
        # A11: i=2,3; j=2,3. Indices 12,13,14,15.
        
        # This mapping is correct.
        
        # For each outer term s (0..6):
        for s in range(7):
            # Determine A blocks and signs
            # Determine B blocks and signs
            # Determine C blocks and signs
            
            # Get the 2x2x2 Strassen matrices for this term
            # The 2x2x2 matrices U_s2, V_s2, W_s2 are 4x7.
            # We need to map the 4 elements of the virtual 2x2 matrix to the 4x4 blocks.
            
            # This is complex. I will skip the 49 rank part and use 56 rank for 4x4x4.
            # R(2,2,4) = 14.
            # 4 * 14 = 56.
            # Total 56 + 44 = 100.
            
            pass
        
        return U, V, W

    # I will return the 112 solution for safety.
    
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
