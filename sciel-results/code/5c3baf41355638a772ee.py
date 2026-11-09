import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 core + 16 for 4x4x1 + 20 for 4x1x5
    rank_core = 49
    rank_441 = 16
    rank_415 = 20
    total_rank = rank_core + rank_441 + rank_415
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Core 4x4x4 using Strassen
    # We need to generate the 49 terms.
    # Helper to add a 2x2x2 Strassen decomposition with specific block mappings.
    def add_strassen_2x2(a_blocks, b_blocks, c_block, offset=0):
        # a_blocks: list of (i, j) indices in 4x4 A for the two 2x2 blocks? 
        # Actually, Strassen operates on 2x2 matrices.
        # a_blocks should be a 2x2 list of 2x2 blocks? No.
        # The inputs to the 2x2 Strassen step are linear combinations.
        # Let's just hardcode the 49 terms by expanding the recursion.
        pass

    # Simpler: Just use the known coefficients for 4x4 Strassen.
    # Or, implement the recursion properly.
    
    # Let's implement the recursion.
    # State: (A_blocks, B_blocks, C_block)
    # A_blocks is a 2x2 array of 2x2 blocks? No, at the bottom level, it's scalars.
    
    # We will generate the terms for 4x4x4.
    # Level 1: 7 terms, each is a 2x2x2 multiplication.
    # Level 2: Each 2x2x2 multiplication has 7 terms, each is a scalar multiplication.
    # Total 49 scalar terms.
    
    # Strassen 2x2 definitions:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4, A is 2x2 blocks of 2x2.
    # A11, A12, A21, A22.
    
    # We need to map the "scalar" multiplications of the bottom level to the global indices.
    
    # Let's define the 7 Strassen steps for the top level.
    # Each step k has:
    # A_input: linear combo of A_blocks
    # B_input: linear combo of B_blocks
    # C_output: linear combo of C_blocks (or intermediate M_k)
    
    # Actually, M_k is an intermediate 2x2 matrix.
    # So C_output of the sub-problem is the matrix M_k.
    # Then we combine M_k to form C.
    
    # This is getting complex to code dynamically.
    # Given the time constraint, I will hardcode the 49 terms if I can, or use a simpler reduction.
    
    # Alternative: Use rank 64 for 4x4x4 (Schoolbook) -> Total Rank 64+16+20 = 100. No improvement.
    # Use rank 49.
    
    # I will write a helper that generates the 49 terms.
    
    # Indices for 4x4:
    # i, j in 0..3.
    # Block (bi, bj) covers i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1].
    
    # Strassen Step Definitions (Top Level):
    # Each step produces a 2x2 matrix M_k.
    # M_k is formed by multiplying X_k and Y_k (2x2 matrices).
    # X_k is linear combo of A_blocks.
    # Y_k is linear combo of B_blocks.
    
    # Step 1: X = A11 + A22, Y = B11 + B22. M1 = X Y.
    # Step 2: X = A21 + A22, Y = B11. M2 = X Y.
    # Step 3: X = A11, Y = B12 - B22. M3 = X Y.
    # Step 4: X = A22, Y = B21 - B11. M4 = X Y.
    # Step 5: X = A11 + A12, Y = B22. M5 = X Y.
    # Step 6: X = A21 - A11, Y = B11 + B12. M6 = X Y.
    # Step 7: X = A12 - A22, Y = B21 + B22. M7 = X Y.
    
    # Then C is formed from M1..M7.
    
    # We need to decompose each M_k = X_k Y_k using Strassen-2x2 (Rank 7).
    # X_k and Y_k are 2x2 matrices.
    # X_k entries are linear combos of A_block entries.
    # Y_k entries are linear combos of B_block entries.
    
    # For each of the 7 steps, we call a function that generates the 7 rank-1 terms for the 2x2 multiplication.
    # The "A" matrix for the sub-problem is X_k.
    # The "B" matrix for the sub-problem is Y_k.
    # The "C" matrix for the sub-problem is M_k.
    
    # How to represent X_k in U?
    # X_k[0,0] = c1 * A[i1,j1] + c2 * A[i2,j2] + ...
    # In U, we add c1 to row (i1,j1), c2 to row (i2,j2), etc.
    
    # Let's implement this.
    
    def get_block_indices(bi, bj):
        # Returns list of (i, j) for the 2x2 block at (bi, bj)
        return [(2*bi + di, 2*bj + dj) for di in [0,1] for dj in [0,1]]
    
    # Strassen 2x2 factors (Rank 7)
    # Returns list of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): coeff} for the 2x2 A matrix
    # V_coeffs: dict {(i,j): coeff} for the 2x2 B matrix
    # W_coeffs: dict {(i,j): coeff} for the 2x2 C matrix
    
    def strassen_2x2_terms():
        terms = []
        # M1
        terms.append((
            {(0,0):1, (1,1):1}, 
            {(0,0):1, (1,1):1}, 
            {(0,0):1, (0,1):0, (1,0):0, (1,1):0} # Placeholder, W is determined by C formula
        ))
        # This is not quite right. The W part is the contribution to C.
        # In the sub-problem, C is M_k.
        # So W_coeffs should be the identity for M_k? No, the sub-problem computes M_k.
        # So the output of the sub-problem is the matrix M_k.
        # In the global tensor, we need to add M_k to C.
        # So the W part of the global term is the position in C where M_k goes.
        
        # Let's restructure.
        # The global term is a scalar multiplication.
        # U_row: linear combo of A indices.
        # V_col: linear combo of B indices.
        # W_row: linear combo of C indices.
        
        # For the sub-problem M_k = X Y:
        # We use Strassen-2x2 on X and Y.
        # This gives 7 scalar terms.
        # For each scalar term t:
        #   U_t: linear combo of X entries.
        #   V_t: linear combo of Y entries.
        #   W_t: linear combo of M_k entries (specifically, which entry of M_k is being computed).
        
        # Then we map X entries to A indices, Y entries to B indices, M_k entries to C indices.
        
        # Strassen 2x2 scalar terms:
        # 1. (X00+X11)(Y00+Y11) -> contributes to M00, M11
        # 2. (X10+X11)Y00 -> contributes to M10, M11
        # 3. X00(Y01-Y11) -> contributes to M01, M11
        # 4. X11(Y10-Y00) -> contributes to M00, M10
        # 5. (X00+X01)Y11 -> contributes to M00, M01
        # 6. (X10-X00)(Y00+Y01) -> contributes to M11
        # 7. (X01-X11)(Y10+Y11) -> contributes to M00, M11
        
        # Let's list the 7 terms with their contributions to C (M_k).
        # Term 1: U={(0,0):1, (1,1):1}, V={(0,0):1, (1,1):1}, W={(0,0):1, (1,1):1}
        # Term 2: U={(1,0):1, (1,1):1}, V={(0,0):1}, W={(1,0):1, (1,1):-1} # Check C11 = M1 - M2 ...
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1 (M1): W={(0,0):1, (1,1):1}
        # Term 2 (M2): W={(1,0):1, (1,1):-1}
        # Term 3 (M3): W={(0,1):1, (1,1):1}
        # Term 4 (M4): W={(0,0):1, (1,0):1}
        # Term 5 (M5): W={(0,0):-1, (0,1):1}
        # Term 6 (M6): W={(1,1):1}
        # Term 7 (M7): W={(0,0):1}
        
        return [
            (
                {(0,0):1, (1,1):1}, 
                {(0,0):1, (1,1):1}, 
                {(0,0):1, (1,1):1}
            ),
            (
                {(1,0):1, (1,1):1}, 
                {(0,0):1}, 
                {(1,0):1, (1,1):-1}
            ),
            (
                {(0,0):1}, 
                {(0,1):1, (1,1):-1}, 
                {(0,1):1, (1,1):1}
            ),
            (
                {(1,1):1}, 
                {(1,0):1, (0,0):-1}, 
                {(0,0):1, (1,0):1}
            ),
            (
                {(0,0):1, (0,1):1}, 
                {(1,1):1}, 
                {(0,0):-1, (0,1):1}
            ),
            (
                {(1,0):1, (0,0):-1}, 
                {(0,0):1, (0,1):1}, 
                {(1,1):1}
            ),
            (
                {(0,1):1, (1,1):-1}, 
                {(1,0):1, (1,1):1}, 
                {(0,0):1}
            )
        ]

    # Now, for each of the 7 top-level steps, we define X and Y in terms of A and B blocks.
    # A blocks: A11, A12, A21, A22.
    # B blocks: B11, B12, B21, B22.
    
    # Step 1: X = A11 + A22, Y = B11 + B22
    # Step 2: X = A21 + A22, Y = B11
    # Step 3: X = A11, Y = B12 - B22
    # Step 4: X = A22, Y = B21 - B11
    # Step 5: X = A11 + A12, Y = B22
    # Step 6: X = A21 - A11, Y = B11 + B12
    # Step 7: X = A12 - A22, Y = B21 + B22
    
    # For each step, we iterate over the 7 Strassen-2x2 terms.
    # For each term, we expand X and Y into global A and B indices.
    # And M_k into global C indices.
    
    base_terms = strassen_2x2_terms()
    
    steps = [
        # (X_coeffs, Y_coeffs, C_block)
        # X_coeffs: dict {block: coeff}
        # Y_coeffs: dict {block: coeff}
        # C_block: (bi, bj)
        ({'A11':1, 'A22':1}, {'B11':1, 'B22':1}, (0,0)), # M1
        ({'A21':1, 'A22':1}, {'B11':1}, (1,0)), # M2
        ({'A11':1}, {'B12':1, 'B22':-1}, (0,1)), # M3
        ({'A22':1}, {'B21':1, 'B11':-1}, (1,1)), # M4
        ({'A11':1, 'A12':1}, {'B22':1}, (0,0)), # M5
        ({'A21':1, 'A11':-1}, {'B11':1, 'B12':1}, (1,1)), # M6
        ({'A12':1, 'A22':-1}, {'B21':1, 'B22':1}, (0,1)) # M7
    ]
    
    # Note: The C_block in the list above is the block where M_k resides?
    # No, M_k is a full 2x2 matrix.
    # The C_block in the Strassen formula determines how M_k is added to C.
    # But in the sub-problem, the output is the full 2x2 matrix M_k.
    # So the W part of the sub-problem is the identity mapping from M_k indices to M_k indices.
    # Then we map M_k indices to C indices.
    
    # Let's refine the loop.
    for x_coeffs, y_coeffs, c_block in steps:
        # c_block is not used directly here, because the sub-problem outputs M_k (2x2).
        # The mapping from M_k to C is handled by the W coefficients in base_terms?
        # No, base_terms W coefficients are for the 2x2 C of the sub-problem.
        # So we just need to map the sub-problem's C (which is M_k) to the global C.
        # The sub-problem's C indices (0,0)..(1,1) correspond to the 2x2 block of C?
        # No, M_k is an intermediate.
        # We need to know where M_k goes in C.
        # From Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for Step 1 (M1):
        # M1[0,0] adds 1 to C[0,0]
        # M1[1,1] adds 1 to C[1,1]
        
        # So we need a mapping for each step: M_k[i,j] -> C[bi+i, bj+j] with coefficient.
        
        m1_map = {(0,0): (0,0, 1), (1,1): (1,1, 1)}
        m2_map = {(1,0): (1,0, 1), (1,1): (1,1, -1)}
        m3_map = {(0,1): (0,1, 1), (1,1): (1,1, 1)}
        m4_map = {(0,0): (0,0, 1), (1,0): (1,0, 1)}
        m5_map = {(0,0): (0,0, -1), (0,1): (0,1, 1)}
        m6_map = {(1,1): (1,1, 1)}
        m7_map = {(0,0): (0,0, 1)}
        
        maps = [m1_map, m2_map, m3_map, m4_map, m5_map, m6_map, m7_map]
        
        for s_idx, (x_c, y_c, _) in enumerate(steps):
            m_map = maps[s_idx]
            for u_t, v_t, w_t in base_terms:
                # Expand U
                u_global = {}
                for (xi, xj), coeff_x in x_c.items():
                    # x_c is dict of block -> coeff
                    # Wait, x_c is like {'A11':1, 'A22':1}
                    pass
                # I need to expand the block coefficients to index coefficients.
                # Let's do this properly.
                pass

    # This is too complex for the time limit.
    # I will return the schoolbook solution to ensure validity, as I cannot verify the Strassen implementation in time.
    # But the score will be 1.0.
    
    # Let's try a simpler improvement: Rank 90?
    # No, 85 is the target.
    
    # I will stick to the schoolbook solution to avoid invalidity.
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
# EVOLVE-BLOCK-END

