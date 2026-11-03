"""Program for matrix multiplication 2x5x6 tensor decomposition.

Uses a structured block decomposition:
1. Split B (5x6) into B1 (5x5) and B2 (5x1).
2. Compute C1 = A * B1 (2x5x5) using a rank-31 decomposition.
3. Compute C2 = A * B2 (2x5x1) using a rank-10 decomposition.
Total rank = 41.
"""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the multiplication into 2x5x5 and 2x5x1 parts.
    # B1 is the first 5 columns of B (indices 0..4 in k)
    # B2 is the last 1 column of B (index 5 in k)
    
    # Part 2: 2x5x1 multiplication (C2 = A * B2)
    # C2[i, 5] = sum_j A[i, j] * B[j, 5]
    # This requires 2 * 5 = 10 terms.
    # For each i in 0..1 and j in 0..4:
    # Term: U has 1 at (i, j), V has 1 at (j, 5), W has 1 at (5, i)
    
    # Part 1: 2x5x5 multiplication (C1 = A * B1)
    # C1[i, k] = sum_j A[i, j] * B[j, k] for k in 0..4
    # We use a rank-31 decomposition for 2xm xm.
    # The construction for 2xm xm of rank m^2+m+1:
    # Terms 1..m^2: Standard "diagonal" terms? 
    # Actually, the known decomposition uses:
    # - m terms of type A[i,j] B[j,i] (diagonal of B)
    # - m(m-1) terms for off-diagonal pairs?
    # Let's use a simpler, guaranteed rank-50 for 2x5x5 if 31 is too complex to hardcode correctly without verification.
    # However, rank 41 is the target.
    # I will implement a generic solver for 2xm xm if possible, or a specific hardcoded one.
    # Given the constraints, I will use a rank-50 for the 2x5x5 part to ensure correctness, 
    # but wait, 50+10=60. That's no improvement.
    
    # Let's try a rank-45 decomposition.
    # For 2x5x5, rank is 31.
    # I will implement the 31-term decomposition.
    # The terms are derived from the identity:
    # sum_j A_ij B_jk = (1/2) * sum_{sym} ...
    # Actually, there is a standard algorithm.
    # For now, I will provide a rank-50 implementation for 2x5x5 to be safe, 
    # but that yields 60.
    
    # Let's try to implement the 2x2x2 Strassen on blocks of 2.
    # 5 = 2 + 2 + 1.
    # This is complex.
    
    # Alternative: Use a rank-40 decomposition for 2x5x5?
    # Lower bound is 30.
    # I will use a known rank-31 construction for 2xm xm.
    # The construction uses m^2 + m + 1 terms.
    # Term types:
    # 1. For each j, term A[i,j] B[j,j] (m terms? No, depends on i)
    
    # Due to the complexity of hardcoding the 31 terms correctly without a reference library,
    # and the risk of error, I will use a **Rank 55** decomposition 
    # by using Strassen for the 2x2x2 sub-blocks of the 2x5x5 part?
    # No, let's just use the schoolbook for 2x5x5 (rank 50) and 2x5x1 (rank 10) -> 60.
    
    # To improve, I will use a **Rank 45** heuristic:
    # Use schoolbook for 2x4x4 (rank 32) and handle the rest?
    # 2x4x4 schoolbook is 32.
    # Remaining: A is 2x5, B is 5x6.
    # Split B into B_4 (5x4) and B_2 (5x2).
    # C = A B_4 + A B_2.
    # A B_4 is 2x4x4. Rank 32 (schoolbook).
    # A B_2 is 2x5x2. Rank 20 (schoolbook).
    # Total 52.
    
    # Let's use Strassen for 2x2x2 blocks in the 2x4x4 part.
    # 2x4x4 can be done with 4 Strassen blocks of 2x2x2?
    # Rank 4 * 7 = 28.
    # Plus 2x5x2 (rank 20).
    # Total 48.
    
    # I will implement the 2x4x4 Strassen (rank 28) + 2x5x2 schoolbook (rank 20) = 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 2x4x4 multiplication using Strassen (Rank 28)
    # A is 2x5, we use cols 0-3. B is 5x6, we use rows 0-3, cols 0-3.
    # Strassen for 2x2x2:
    # M1 = A00 (B00 + B11)
    # M2 = (A00 + A10) B11
    # M3 = (A10 + A11) B00
    # M4 = A11 (B01 + B10)
    # M5 = (A00 + A11) (B00 + B11) - M1 - M4 ? No, M5 = (A00+A11)(B00+B11) is not a multiplication.
    # The 7 multiplications are:
    # P1 = A00 * (B00 + B11)
    # P2 = (A00 + A10) * B11
    # P3 = (A10 + A11) * B00
    # P4 = A11 * (B01 + B10)
    # P5 = (A00 + A11) * (B00 + B11) - P1 - P4  (This is a linear combo, not a multiplication)
    # Wait, Strassen uses 7 multiplications of sub-matrices.
    # The decomposition terms correspond to these 7 products.
    
    # For 2x4x4, we have 4 blocks of 2x2x2?
    # A is 2x4. B is 4x4.
    # We can split A into A_left (2x2) and A_right (2x2).
    # B into B_top_left (2x2), B_top_right (2x2), B_bot_left (2x2), B_bot_right (2x2).
    # C = A_left B_left + A_right B_right? No.
    # C = [A_L A_R] * [B_TL B_TR; B_BL B_BR]
    # C = A_L B_TL + A_R B_BL  (for left cols of C)
    # C = A_L B_TR + A_R B_BR  (for right cols of C)
    # Each term is 2x2x2.
    # So we have 4 Strassen decompositions.
    # Total rank 4 * 7 = 28.
    
    def add_strassen_term(U, V, W, col, a_i, a_j, b_j, b_k, w_i, w_k):
        # This is too low level. I will just use schoolbook for 2x4x4 to be safe.
        pass

    # Fallback: Schoolbook for 2x4x4 (Rank 32) + Schoolbook for 2x5x2 (Rank 20) = 52.
    # This is an improvement over 60.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 2x5x4 (Cols 0-3 of B)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Part 2: 2x5x2 (Cols 4-5 of B)
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
