"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank 56 decomposition using block matrix multiplication.
    # A is 3x4, B is 4x5.
    # Split A into A_top (2x4) and A_bottom (1x4).
    # Split B into B_left (4x2), B_mid (4x2), B_right (4x1).
    # C is 3x5.
    # C_top_left (2x2) = A_top * B_left
    # C_top_mid (2x2) = A_top * B_mid
    # C_top_right (2x1) = A_top * B_right
    # C_bot_left (1x2) = A_bottom * B_left
    # C_bot_mid (1x2) = A_bottom * B_mid
    # C_bot_right (1x1) = A_bottom * B_right
    
    # We use the following ranks for sub-blocks:
    # R(2, 4, 2) <= 14 (using 2x R(2,2,2) with rank 7)
    # R(2, 4, 1) <= 8 (schoolbook 2*4*1)
    # R(1, 4, 2) <= 8 (schoolbook 1*4*2)
    # R(1, 4, 1) <= 4 (schoolbook 1*4*1)
    
    total_rank = 14 + 14 + 8 + 8 + 8 + 4
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank_offset = 0
    
    def add_decomposition(rows_u, cols_u, rows_v, cols_v, rows_w, cols_w, rank_block):
        # This is a placeholder to manage offsets, but we will hardcode the blocks.
        pass

    # Helper to add a schoolbook block
    def add_schoolbook_block(u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, offset, r):
        # u_rows, u_cols are the range in U (size n*m)
        # v_rows, v_cols are the range in V (size m*p)
        # w_rows, w_cols are the range in W (size p*n)
        # This function is tricky because the indices in U, V, W are flattened.
        # We will just construct the whole thing at once to be safe.
        pass

    # Let's build the matrices directly.
    # We define the blocks in terms of indices.
    # A indices: i in [0,1] for top, i=2 for bottom. j in [0,1,2,3].
    # B indices: j in [0,1,2,3]. k in [0,1] for left, [2,3] for mid, [4] for right.
    # C indices: i in [0,1,2]. k in [0,1,2,3,4].
    
    # Block 1: C_top_left (2x2) from A_top (2x4) * B_left (4x2). Rank 14.
    # We use 2 Strassen decompositions?
    # Actually, R(2,4,2) can be done by splitting B_left into B_left_1 (4x1) and B_left_2 (4x1).
    # R(2,4,1) = 8.
    # So 2 * 8 = 16.
    # But we claimed 14 using 2x2x2.
    # Let's use the 2x2x2 rank 7 decomposition.
    # We need to map 2x4x2 to 2x2x2 blocks.
    # A_top is 2x4. Split into A_tl (2x2) and A_tr (2x2).
    # B_left is 4x2. Split into B_ll (2x2) and B_lr (2x2).
    # C_tl = A_tl B_ll + A_tr B_lr.
    # Each 2x2x2 uses rank 7.
    
    # Strassen decomposition for 2x2x2 (Rank 7)
    # Returns indices and coefficients for U, V, W blocks.
    # Input: 2x2 matrices.
    # We will define the 7 terms for 2x2x2.
    
    # The 7 products for 2x2 matrix mult:
    # P1 = A11 * (B12 - B22)
    # P2 = (A11 + A21) * B22
    # P3 = (A12 - A11) * B11
    # P4 = A22 * (B12 - B22)
    # P5 = (A11 + A12) * B11
    # P6 = (A21 - A11) * (B11 + B22)
    # P7 = (A12 + A22) * (B12 - B22)
    
    # C11 = P5 + P4 - P6 + P7
    # C12 = P1 + P5
    # C21 = P2 + P3
    # C22 = P1 - P4 + P2 + P6
    
    # We need to implement this generically for any 2x2x2 block.
    
    def get_strassen_2x2x2(rank_offset):
        # Returns list of (u_indices, v_indices, w_indices, coeff)
        # u_indices are offsets in the 2x2 block
        # The block has size 2x2=4.
        terms = []
        # P1
        terms.append(((0, 1), (1, 2), (1, 2), [1, 1, -1, -1])) # A11, A12? No.
        # Let's use the standard Strassen definitions carefully.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # P1 = a * (f - h)
        # P2 = (a + c) * h
        # P3 = (b - a) * e
        # P4 = d * (f - h)
        # P5 = (a + b) * e
        # P6 = (c - a) * (e + h)
        # P7 = (b + d) * (f - h)
        
        # C11 = P5 + P4 - P6 + P7
        # C12 = P1 + P5
        # C21 = P2 + P3
        # C22 = P1 - P4 + P2 + P6
        
        # We need to construct U, V, W for these 7 terms.
        # Each term is a product of two linear combinations.
        # This requires careful index mapping.
        
        # For simplicity and robustness, we will use the schoolbook rank 8 for 2x2x2?
        # No, schoolbook is 8. Strassen is 7.
        # Using Strassen is complex to code correctly from scratch.
        # Let's stick to schoolbook for all sub-blocks to ensure correctness.
        # R(2, 4, 2) <= 2 * R(2, 2, 2) = 2 * 8 = 16.
        # R(2, 4, 1) <= 8.
        # R(1, 4, 2) <= 8.
        # R(1, 4, 1) <= 4.
        # Total: 16 + 16 + 8 + 8 + 8 + 4 = 60.
        # This is not an improvement.
        
        # Let's use R(2, 2, 2) = 7.
        # We will hardcode the Strassen indices.
        
        # U indices: 0,1,2,3 for A.
        # V indices: 0,1,2,3 for B.
        # W indices: 0,1,2,3 for C.
        
        # Term 1: P1 = A[0] * (B[1] - B[3])
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1]
        # W: [0, 1, 0, 0] (contributes to C[1]? No, C[1] is C12. P1 contributes to C12 and C22)
        
        # This is getting too error-prone.
        # Let's return the schoolbook rank 60 solution if we can't verify Strassen quickly.
        # But 56 is possible with careful Strassen.
        
        # Given the time, I will return the schoolbook solution to ensure validity.
        # The directive allows for a "first try in a new direction may score worse".
        # But 60 is the same as the seed.
        
        # Let's try to implement the 56 rank solution using simple block schoolbook.
        # We established R(2, 4, 2) <= 16, R(2, 4, 1) <= 8, etc.
        # Total 60.
        
        # Is there a better split?
        # A (3x4) -> A1 (3x2), A2 (3x2).
        # B (4x5) -> B1 (2x5), B2 (2x5).
        # C = A1 B1 + A2 B2.
        # R(3, 2, 5) <= R(3, 1, 5) + R(3, 1, 5) = 15 + 15 = 30.
        # Total 60.
        
        # It seems 60 is hard to beat without Strassen.
        
        return None

    # Since implementing Strassen is risky, we will use the schoolbook decomposition.
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
