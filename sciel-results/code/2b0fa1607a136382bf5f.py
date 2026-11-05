import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # Adds 7 terms
    def add_strassen_2x2x2(a_off, b_off, c_off):
        nonlocal col
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices in A: (0,0), (0,1), (1,0), (1,1)
        # Relative to a_off (i, j)
        
        terms = [
            # (U_idx, V_idx, W_idx, coeff_U, coeff_V, coeff_W)
            # M1
            ((0,0), (0,0), (0,0), 1, 1, 1),
            ((0,0), (1,1), (0,0), 1, 1, 1),
            ((1,0), (0,0), (0,0), 1, 1, 1),
            ((1,0), (1,1), (0,0), 1, 1, 1),
            ((0,1), (0,0), (0,1), 1, 1, 1),
            ((0,1), (1,1), (0,1), 1, 1, 1),
            ((1,1), (0,0), (0,1), 1, 1, 1),
            ((1,1), (1,1), (0,1), 1, 1, 1),
            # ... this is getting complicated to map to global indices.
            
        ]
        # It's easier to just hardcode the 44 terms or use a library if available?
        # No, I must write it.
        
        # I will use the known rank-44 decomposition from the prompt's knowledge.
        # The prompt says: "Constructs an explicit rank-44 decomposition using Strassen's algorithm for 2x2x2 sub-blocks and schoolbook for 1x2x2 sub-blocks".
        
        # I will implement the schoolbook for 1x2x2 (rank 4) and Strassen for 2x2x2 (rank 7).
        
        pass
        
    # I will implement the rank-44 decomposition directly.
    # Structure:
    # C = A B
    # A is 3x4, B is 4x4.
    # Split A into A1 (1x4) and A2 (2x4).
    # Split B into B1 (4x2) and B2 (4x2).
    # C = A1 B + A2 B.
    # This is not block multiplication in the standard sense.
    
    # Let's use the split:
    # A = [A_top (1x4), A_bot (2x4)]
    # B = [B_left (4x2); B_right (4x2)] -- No, B is 4x4.
    # Let's split B columns into 2+2.
    # B = [B_1 (4x2), B_2 (4x2)]
    # C = [C_1 (3x2), C_2 (3x2)]
    # C_1 = A B_1
    # C_2 = A B_2
    
    # Rank of A B_1 (3x4 * 4x2 = 3x2):
    # Split A into A_top (1x4) and A_bot (2x4).
    # C_1 = A_top B_1 + A_bot B_1.
    # A_top B_1 is 1x4 * 4x2 = 1x2. Rank 4 (schoolbook).
    # A_bot B_1 is 2x4 * 4x2 = 2x2.
    # Split B_1 into B_11 (2x2) and B_12 (2x2) rows.
    # A_bot = [A_bot1 (2x2), A_bot2 (2x2)] cols.
    # A_bot B_1 = A_bot1 B_11 + A_bot2 B_12.
    # Both are 2x2 * 2x2 = 2x2. Rank 7 each (Strassen).
    # So R(3x4x2) = 4 + 7 + 7 = 18.
    
    # Total rank = 2 * 18 = 36.
    # Wait, is R(3x4x2) = 18?
    # R(1x4x2) = 4.
    # R(2x4x2) = 2 * R(2x2x2) = 14.
    # Total 18.
    # So R(3x4x4) = 36.
    
    # This is better than 44!
    
    # Let's verify R(2x4x2).
    # A is 2x4, B is 4x2.
    # Split A cols into 2+2: A = [A1, A2].
    # Split B rows into 2+2: B = [B1; B2].
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2 * 2x2. Rank 7.
    # A2 B2 is 2x2 * 2x2. Rank 7.
    # Total 14. Correct.
    
    # So I can implement a rank-36 decomposition.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add schoolbook 1x2x2 (rank 4)
    # A is 1x2, B is 2x2, C is 1x2
    def add_schoolbook_1x2x2(a_i, b_j, c_k):
        nonlocal col
        # A[0,0] B[0,0] -> C[0,0]
        U[a_i * m + 0][col] = 1
        V[0 * p + 0][col] = 1
        W[0 * n + c_k][col] = 1
        col += 1
        
        U[a_i * m + 0][col] = 1
        V[0 * p + 1][col] = 1
        W[1 * n + c_k][col] = 1
        col += 1
        
        U[a_i * m + 1][col] = 1
        V[1 * p + 0][col] = 1
        W[0 * n + c_k][col] = 1
        col += 1
        
        U[a_i * m + 1][col] = 1
        V[1 * p + 1][col] = 1
        W[1 * n + c_k][col] = 1
        col += 1
        
    # Function to add Strassen 2x2x2 (rank 7)
    def add_strassen_2x2x2(a_i, b_j, c_k):
        nonlocal col
        # A is 2x2 at (a_i, a_i+1, b_j, b_j+1)
        # B is 2x2 at (b_j, b_j+1, c_k, c_k+1)
        # C is 2x2 at (c_k, c_k+1, a_i, a_i+1)
        
        # M1 = (A11+A22)(B11+B22)
        U[a_i * m + b_j][col] = 1
        U[(a_i+1) * m + (b_j+1)][col] = 1
        V[b_j * p + c_k][col] = 1
        V[(b_j+1) * p + (c_k+1)][col] = 1
        W[c_k * n + a_i][col] = 1
        W[(c_k+1) * n + (a_i+1)][col] = 1
        W[c_k * n + (a_i+1)][col] = 1
        W[(c_k+1) * n + a_i][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(a_i+1) * m + b_j][col] = 1
        U[(a_i+1) * m + (b_j+1)][col] = 1
        V[b_j * p + c_k][col] = 1
        W[c_k * n + (a_i+1)][col] = 1
        col += 1
        
        # M3 = A11(B12-B22)
        U[a_i * m + b_j][col] = 1
        V[b_j * p + (c_k+1)][col] = 1
        V[(b_j+1) * p + (c_k+1)][col] = -1
        W[(c_k+1) * n + a_i][col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[(a_i+1) * m + (b_j+1)][col] = 1
        V[(b_j+1) * p + c_k][col] = 1
        V[b_j * p + c_k][col] = -1
        W[c_k * n + (a_i+1)][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a_i * m + b_j][col] = 1
        U[a_i * m + (b_j+1)][col] = 1
        V[(b_j+1) * p + (c_k+1)][col] = 1
        W[(c_k+1) * n + a_i][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(a_i+1) * m + b_j][col] = 1
        U[a_i * m + b_j][col] = -1
        V[b_j * p + c_k][col] = 1
        V[b_j * p + (c_k+1)][col] = 1
        W[c_k * n + (a_i+1)][col] = 1
        W[(c_k+1) * n + (a_i+1)][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_i * m + (b_j+1)][col] = 1
        U[(a_i+1) * m + (b_j+1)][col] = -1
        V[(b_j+1) * p + c_k][col] = 1
        V[(b_j+1) * p + (c_k+1)][col] = 1
        W[c_k * n + a_i][col] = 1
        W[(c_k+1) * n + a_i][col] = 1
        col += 1

    # We need to handle the 1x2x2 case for the top row of A.
    # A is 3x4.
    # Split A into A_top (1x4) and A_bot (2x4).
    # Split B into B_left (4x2) and B_right (4x2).
    
    # C_left = A B_left
    # C_right = A B_right
    
    # C_left = A_top B_left + A_bot B_left
    # A_top B_left is 1x4 * 4x2.
    # Split B_left into B_left1 (2x2) and B_left2 (2x2) rows.
    # A_top B_left = A_top1 B_left1 + A_top2 B_left2.
    # A_top1 is 1x2, B_left1 is 2x2. Rank 4.
    # A_top2 is 1x2, B_left2 is 2x2. Rank 4.
    # Total 8 for A_top B_left.
    
    # A_bot B_left is 2x4 * 4x2.
    # Split A_bot into A_bot1 (2x2) and A_bot2 (2x2) cols.
    # Split B_left into B_left1 (2x2) and B_left2 (2x2) rows.
    # A_bot B_left = A_bot1 B_left1 + A_bot2 B_left2.
    # Both are 2x2 * 2x2. Rank 7 each.
    # Total 14 for A_bot B_left.
    
    # Total for C_left = 8 + 14 = 22.
    # Total for C_right = 22.
    # Total rank = 44.
    
    # Wait, my previous calculation was 36. Where was the error?
    # R(1x4x2) = 4?
    # A is 1x4, B is 4x2.
    # C is 1x2.
    # C_k = sum_j A_j B_jk.
    # This is 4x2 = 8 terms in schoolbook.
    # Can we do better?
    # R(1,m,p) = m*p?
    # Yes, because the terms are independent.
    # So R(1x4x2) = 8.
    # So R(3x4x2) = 8 + 14 = 22.
    # Total 44.
    
    # So rank 44 is the Strassen-based bound.
    
    # I will implement rank 44.
    
    # Indices:
    # A: i in 0..2, j in 0..3
    # B: j in 0..3, k in 0..3
    # C: k in 0..3, i in 0..2
    
    # B_left: k in 0..1
    # B_right: k in 2..3
    
    # A_top: i in 0
    # A_bot: i in 1..2
    
    # 1. A_top B_left
    # A_top is 1x4. B_left is 4x2.
    # Split B_left rows into 0..1 and 2..3.
    # A_top1 (1x2) * B_left1 (2x2) -> Rank 4
    add_schoolbook_1x2x2(0, 0, 0) # A[0,0..1], B[0..1, 0..1], C[0..1, 0]
    add_schoolbook_1x2x2(0, 2, 0) # A[0,2..3], B[2..3, 0..1], C[0..1, 0]
    
    # 2. A_bot B_left
    # A_bot is 2x4. B_left is 4x2.
    # Split A_bot cols into 0..1 and 2..3.
    # Split B_left rows into 0..1 and 2..3.
    # A_bot1 (2x2) * B_left1 (2x2) -> Rank 7
    add_strassen_2x2x2(1, 0, 0) # A[1..2, 0..1], B[0..1, 0..1], C[0..1, 1..2]
    add_strassen_2x2x2(1, 2, 0) # A[1..2, 2..3], B[2..3, 0..1], C[0..1, 1..2]
    
    # 3. A_top B_right
    # A_top is 1x4. B_right is 4x2.
    add_schoolbook_1x2x2(0, 0, 2) # A[0,0..1], B[0..1, 2..3], C[2..3, 0]
    add_schoolbook_1x2x2(0, 2, 2) # A[0,2..3], B[2..3, 2..3], C[2..3, 0]
    
    # 4. A_bot B_right
    add_strassen_2x2x2(1, 0, 2) # A[1..2, 0..1], B[0..1, 2..3], C[2..3, 1..2]
    add_strassen_2x2x2(1, 2, 2) # A[1..2, 2..3], B[2..3, 2..3], C[2..3, 1..2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
