import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # We want to decompose the 3x5x6 matrix multiplication tensor.
    # Strategy: Pad to 4x8x8 and decompose using two independent 4x4x4 Strassen decompositions.
    # R(4,4,4) = 14. So total rank = 14 + 14 = 28.
    
    n, m, p = 3, 5, 6
    
    # Dimensions of the padded problem:
    # A (3x5) -> A' (4x8)
    # B (5x6) -> B' (8x8)
    # C (3x6) -> C' (4x8)
    
    # Embedding:
    # A'[i, j] = A[i, j] for i < 3, j < 5; 0 otherwise.
    # B'[j, k] = B[j, k] for j < 5, k < 6; 0 otherwise.
    # C'[i, k] = C[i, k] for i < 3, k < 6; 0 otherwise.
    
    # Decomposition of 4x8x8:
    # A' = [A1, A2] where A1, A2 are 4x4.
    # B' = [B1; B2] where B1, B2 are 4x8? No, B' is 8x8.
    # Let's partition B' into two 4x8 blocks: B_top (rows 0-3), B_bot (rows 4-7).
    # C' = A' B' = A1 B_top + A2 B_bot.
    # So we need to decompose two 4x4x8 multiplications.
    # R(4,4,8) <= 2 * R(4,4,4) = 28.
    # Total rank = 28 + 28 = 56.
    
    # Wait, I can do better.
    # R(4,4,8) is actually 28? Yes, 2 * 14.
    # Is there a better way?
    # R(3,5,6) <= 28 + 28 = 56.
    
    # Let's check if we can do better with a different padding.
    # Pad to 4x6x6?
    # R(4,6,6) <= R(4,4,6) + R(0,2,6) + ...
    # R(4,6,6) <= 2 * R(4,3,3)? No.
    # R(4,6,6) <= R(4,4,4) + R(4,2,2) + R(4,4,2) + R(4,2,4)?
    # This is getting complicated.
    
    # Let's stick with the 4x8x8 embedding. Rank 56 is a valid improvement over 90.
    
    rank = 56
    U = [[0] * rank for _ in range(4 * 8)]  # 32 rows
    V = [[0] * rank for _ in range(8 * 8)]  # 64 rows
    W = [[0] * rank for _ in range(8 * 4)]  # 32 rows
    
    # Strassen decomposition for 4x4x4.
    # We need the 14 terms of the optimal 4x4x4 decomposition.
    # The optimal 4x4x4 decomposition can be constructed from the 2x2x2 Strassen decomposition.
    # 4x4 = 2x(2x2).
    # We can view A, B, C as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A B
    # We can compute C using 7 multiplications of 2x2 matrices? No, that's not how it works.
    # The 4x4x4 tensor has rank 14.
    # One way to get it is to use the "Strassen" method on the 4x4 matrices by splitting them into 2x2 blocks.
    # This requires 7 multiplications of 2x2x2? No.
    
    # Let's use a known 14-term decomposition.
    # The 14 terms can be found in literature.
    # Alternatively, we can use the 7-term Strassen for 2x2x2 and apply it to the 4x4 blocks?
    # No, that would give rank 7 * 4 = 28? No.
    
    # Let's use the schoolbook for 4x4x4 (rank 16) if I can't find the 14-term one?
    # Then total rank = 16 + 16 = 32. Still better than 56? No, 32 < 56.
    # Wait, 16 < 14? No, 16 > 14.
    # So 14 is better.
    
    # I will implement the 14-term 4x4x4 decomposition.
    # Reference: "The rank of the 4x4x4 matrix multiplication tensor is 14"
    # The 14 terms are:
    # 1. A11 B11
    # 2. A22 B22
    # 3. (A11 + A22) (B11 + B22)
    # 4. A12 B21
    # 5. A21 B12
    # 6. (A12 + A22) (B21 + B11)
    # 7. (A11 + A12) (B22 + B21)
    # ... this is for 2x2x2.
    
    # For 4x4x4, we can use the 2x2x2 Strassen on the 4x4 matrices by viewing them as 2x2 blocks of 2x2 matrices.
    # This gives rank 7 * 2 = 14? No.
    # It gives rank 14.
    
    # Let's just use the 16-term schoolbook for 4x4x4 to be safe and simple.
    # Rank = 16 + 16 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(4 * 8)]
    V = [[0] * rank for _ in range(8 * 8)]
    W = [[0] * rank for _ in range(8 * 4)]
    
    # Block 1: A1 (4x4) * B_top (4x8) -> C1 (4x8)
    # Schoolbook: 4*4*8 = 128 terms? No, that's too many.
    # We need to decompose 4x4x8.
    # R(4,4,8) <= 2 * R(4,4,4) = 2 * 16 = 32 (using schoolbook for 4x4x4).
    # So total rank = 32 + 32 = 64.
    
    # This is worse than 90? No, 64 < 90.
    # So rank 64 is valid.
    
    # Let's implement the 64-rank decomposition.
    rank = 64
    U = [[0] * rank for _ in range(4 * 8)]
    V = [[0] * rank for _ in range(8 * 8)]
    W = [[0] * rank for _ in range(8 * 4)]
    
    # Helper to add a schoolbook block
    def add_schoolbook_block(u_offset, v_offset, w_offset, u_rows, v_cols, w_rows, w_cols):
        # u_rows: list of row indices for U (4x4 block)
        # v_cols: list of col indices for V (4x8 block)
        # w_rows: list of row indices for W (4x8 block)
        # w_cols: list of col indices for W (4x8 block)
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    u_idx = u_offset + i * 4 + j
                    v_idx = v_offset + j * 8 + k
                    w_idx = w_offset + k * 4 + i
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return col
    
    # Block 1: A1 (rows 0-3, cols 0-3 of A') * B_top (rows 0-3, cols 0-7 of B') -> C1 (rows 0-3, cols 0-7 of C')
    # A' is 4x8. A1 is the left 4x4 block.
    # B' is 8x8. B_top is the top 4x8 block.
    # C' is 4x8. C1 is the left 4x4 block? No, C' = A1 B_top + A2 B_bot.
    # So C1 is the left 4x4 block of C'? No, C' is 4x8.
    # C' = [C_left, C_right].
    # C_left = A1 B_top_left + A2 B_bot_left.
    # C_right = A1 B_top_right + A2 B_bot_right.
    
    # So we have 4 blocks:
    # 1. A1 (4x4) * B_top_left (4x4) -> C_left (4x4)
    # 2. A1 (4x4) * B_top_right (4x4) -> C_right (4x4)
    # 3. A2 (4x4) * B_bot_left (4x4) -> C_left (4x4)
    # 4. A2 (4x4) * B_bot_right (4x4) -> C_right (4x4)
    
    # Each block is 4x4x4. Schoolbook rank 16.
    # Total rank = 4 * 16 = 64.
    
    # U indices for A1: rows 0-3, cols 0-3.
    # U indices for A2: rows 0-3, cols 4-7.
    # V indices for B_top_left: rows 0-3, cols 0-3.
    # V indices for B_top_right: rows 0-3, cols 4-7.
    # V indices for B_bot_left: rows 4-7, cols 0-3.
    # V indices for B_bot_right: rows 4-7, cols 4-7.
    # W indices for C_left: rows 0-3, cols 0-3.
    # W indices for C_right: rows 0-3, cols 4-7.
    
    # Note: W is 8x4? No, W is 8x4 in the code above?
    # W shape is (p*n, rank) = (6*3, rank) = (18, rank)?
    # No, the interface says W is (p*n, rank).
    # But in the embedding, C' is 4x8.
    # So W should be 4*8 = 32 rows?
    # The interface says W is (p*n, rank) = (18, rank).
    # This is a mismatch.
    
    # Let's re-read the interface.
    # U: (n*m, rank) = (15, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (18, rank)
    
    # So we cannot use the 4x8x8 embedding directly because the dimensions don't match.
    # We must map the 4x8x8 decomposition back to the 3x5x6 problem.
    
    # This is getting too complex. Let's just use the schoolbook for 3x5x6 (rank 90).
    
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
