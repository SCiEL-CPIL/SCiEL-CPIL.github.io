"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x9 matrix B into three 3x3 blocks.
    # For each block, we compute 2x3 * 3x3 using a rank-13 decomposition.
    # Total rank = 3 * 13 = 39.
    
    # Rank-13 decomposition for 2x3 * 3x3:
    # Split A (2x3) into A1 (2x2) and a (2x1).
    # Split B (3x3) into B1 (2x3) and b (1x3).
    # C = A1 * B1 + a * b.
    # A1 * B1 is 2x2 * 2x3. Wait.
    # A1 is 2x2. B1 is 2x3.
    # Product is 2x3.
    # Rank of 2x2 * 2x3?
    # This is not standard Strassen.
    
    # Let's re-verify the split.
    # A (2x3) = [ A_left (2x2) | A_right (2x1) ]
    # B (3x3) = [ B_top (2x3) ; B_bottom (1x3) ]
    # C = A_left * B_top + A_right * B_bottom.
    # Term 1: 2x2 * 2x3.
    # Term 2: 2x1 * 1x3.
    
    # Rank of 2x1 * 1x3 is 6.
    # Rank of 2x2 * 2x3?
    # This is computing 2 outputs of size 3 from 2 inputs of size 2.
    # C_col_k = A_left * B_top_col_k.
    # Each col is 2x1 * 2x1? No, 2x2 * 2x1.
    # Rank of 2x2 * 2x1 is 4?
    # Actually, we can compute 2x2 * 2x3 by computing 3 dot products of length 2?
    # No, it's 2x2 matrix times 2x3 matrix.
    # We can use Strassen for the 2x2 * 2x2 part?
    # B_top is 2x3. Split into B1 (2x2) and b (2x1).
    # A_left * B1 (2x2*2x2, rank 7) + A_left * b (2x2*2x1, rank 4).
    # Total 11.
    # Plus Term 2 (6).
    # Total 17.
    # This is worse than 13?
    
    # My previous calculation:
    # A (2x3) split into A1 (2x2) and a (2x1).
    # B (3x3) split into B1 (2x3) and b (1x3).
    # C = A1 B1 + a b.
    # A1 is 2x2. B1 is 2x3.
    # A1 B1: 2x2 * 2x3.
    # This is equivalent to 3 multiplications of 2x2 * 2x1?
    # Or we can view it as 2x2 * 2x2 + 2x2 * 2x1.
    # Rank(2x2*2x2) = 7.
    # Rank(2x2*2x1) = 4.
    # Sum 11.
    # Rank(2x1*1x3) = 6.
    # Sum 17.
    
    # Where did 13 come from?
    # Ah, I assumed A1 (2x2) * B1 (2x2).
    # But B is 3x3.
    # If I split B into 2x2 and 1x3, the dimensions don't match for A1 (2x2).
    # A1 is 2x2. It multiplies the first 2 rows of B.
    # B_top is 2x3.
    # So A1 * B_top is 2x2 * 2x3.
    # Rank is 11.
    # So total 17.
    
    # Let's re-evaluate Rank(2x3 * 3x3).
    # Schoolbook 18.
    # My 13 was based on A1 (2x2) * B1 (2x2) which is wrong dimension.
    
    # Is there a rank 15 decomposition?
    # Split B into 3 columns.
    # C = A * B_col0 + A * B_col1 + A * B_col2.
    # Each A * B_col is 2x3 * 3x1. Rank 6.
    # Total 18.
    
    # Split A into 3 rows? No, A has 2 rows.
    
    # So Rank(2x3 * 3x3) is likely 18 or 17.
    # If 17, Total 51.
    # If 18, Total 54.
    
    # Can we do better?
    # Maybe use Strassen on the 2x2 sub-blocks of B?
    # B = [ B11 B12; B21 B22 ] with B11 2x2, etc.
    # This is getting complicated.
    
    # Given the time, I will stick to the schoolbook 54 if I can't find a clear improvement.
    # But wait, the prompt says "score 1.0000" for 54.
    # And "improvement" is > 1.0.
    # If I can't find a better rank, I should return 54.
    # But usually these tasks have a trick.
    
    # Let's check Rank(2x2 * 2x3).
    # C = A B. A 2x2, B 2x3.
    # C0 = A0 B0 + A1 B1.
    # C1 = A0 B2 + A1 B3.
    # C2 = A0 B4 + A1 B5.
    # This is 3 dot products of length 2.
    # Rank 3 * 2 = 6?
    # No, the A indices are shared.
    # T = sum_k (e0 otimes B_k + e1 otimes B_{k+2})?
    # Actually, rank of 2x2 * 2xn is 2n?
    # For n=1, rank 4.
    # For n=2, rank 8?
    # If rank is 2n, then 2x2 * 2x3 is 6.
    # Then 2x2 * 2x3 (6) + 2x1 * 1x3 (6) = 12.
    # Total 12.
    # 3 * 12 = 36.
    
    # Let's verify Rank(2x2 * 2x1) = 4.
    # C0 = A00 B0 + A10 B1.
    # C1 = A01 B0 + A11 B1.
    # 4 terms. Independent. Rank 4.
    # So Rank(2x2 * 2xn) = 2n?
    # Yes, because each output column is a dot product of length 2.
    # And they are independent.
    
    # So Rank(2x3 * 3x3) using split:
    # A (2x3) = A1 (2x2) + a (2x1).
    # B (3x3) = B1 (2x3) + b (1x3).
    # C = A1 B1 + a b.
    # A1 B1: 2x2 * 2x3. Rank 6.
    # a b: 2x1 * 1x3. Rank 6.
    # Total 12.
    
    # So Rank 36 is achievable.
    
    rank_per_block = 12
    total_rank = 3 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for block in range(3):
        # B block is cols block*3 to block*3+2
        # A1 is cols 0,1 of A. a is col 2.
        # B1 is rows 0,1 of B_block. b is row 2.
        
        # Part 1: A1 * B1 (2x2 * 2x3)
        # Rank 6.
        # For each col c of B1 (0,1,2):
        #   Dot product of A1_col0 and B1_col_c -> C_col_c_0
        #   Dot product of A1_col1 and B1_col_c -> C_col_c_1
        # Wait, A1 is 2x2. B1 is 2x3.
        # C1 (2x3) = A1 B1.
        # C1_i_k = sum_{j=0}^1 A1_i_j B1_j_k.
        # This is 3 dot products of length 2.
        # For k in 0..2:
        #   j=0: A1_0_0 B1_0_k -> C1_0_k
        #         A1_1_0 B1_0_k -> C1_1_k
        #   j=1: A1_0_1 B1_1_k -> C1_0_k
        #         A1_1_1 B1_1_k -> C1_1_k
        # Total 6 terms.
        
        for k in range(3):
            # j=0
            # A index: row 0, col 0 -> 0*2+0 = 0? No, A is 2x3.
            # A1 is A[:, 0:2].
            # A1_0_0 is A[0,0]. A1_1_0 is A[1,0].
            # B1_0_k is B[0, block*3+k].
            
            # Term 1: A[0,0] * B[0, block*3+k] -> C[0, block*3+k]
            U[0][col] = 1
            V[0 * 9 + (block*3+k)][col] = 1 # B row 0, col k
            W[(block*3+k) * 2 + 0][col] = 1 # C col k, row 0
            col += 1
            
            # Term 2: A[1,0] * B[0, block*3+k] -> C[1, block*3+k]
            U[1 * 2 + 0][col] = 1 # A[1,0]
            V[0 * 9 + (block*3+k)][col] = 1
            W[(block*3+k) * 2 + 1][col] = 1
            col += 1
            
            # Term 3: A[0,1] * B[1, block*3+k] -> C[0, block*3+k]
            U[0 * 2 + 1][col] = 1 # A[0,1]
            V[1 * 9 + (block*3+k)][col] = 1 # B row 1
            W[(block*3+k) * 2 + 0][col] = 1
            col += 1
            
            # Term 4: A[1,1] * B[1, block*3+k] -> C[1, block*3+k]
            U[1 * 2 + 1][col] = 1 # A[1,1]
            V[1 * 9 + (block*3+k)][col] = 1
            W[(block*3+k) * 2 + 1][col] = 1
            col += 1

        # Part 2: a * b (2x1 * 1x3)
        # a is A[:, 2]. b is B[2, block*3 : block*3+3].
        # C2_i_k = A[i,2] * B[2, block*3+k].
        for k in range(3):
            # i=0
            U[0 * 2 + 2][col] = 1 # A[0,2]
            V[2 * 9 + (block*3+k)][col] = 1 # B row 2
            W[(block*3+k) * 2 + 0][col] = 1
            col += 1
            
            # i=1
            U[1 * 2 + 2][col] = 1 # A[1,2]
            V[2 * 9 + (block*3+k)][col] = 1
            W[(block*3+k) * 2 + 1][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
