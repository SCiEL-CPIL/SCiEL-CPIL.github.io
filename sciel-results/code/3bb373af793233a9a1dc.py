import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition:
    # Split B into B1 (3x2) and B2 (3x3)
    # C1 = A B1 (2x2), Rank 12 (Schoolbook)
    # C2 = A B2 (2x3), Rank 14 (Optimal)
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Schoolbook for 2x3 x 3x2 (Rank 12)
    # A is 2x3, B1 is 3x2 (cols 0,1 of B)
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(2): # k is 0,1
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
                
    # Part 2: Rank 14 for 2x3 x 3x3 (cols 2,3,4 of B)
    # This part requires the 14-term decomposition.
    # I will use a known decomposition for 2x3x3.
    # Since I don't have it hardcoded, I will use Schoolbook (Rank 18) for this part.
    # Total Rank = 12 + 18 = 30.
    # This is not an improvement.
    
    # I must find the Rank 14 decomposition.
    # If I cannot, I will return Rank 30.
    # But I need to improve.
    
    # Let's try Rank 28 by using Strassen for 2x2x2 part of the 2x3x3?
    # No, I will just return the seed for now if I can't find the exact terms.
    # But the prompt asks for a rewrite.
    
    # I will implement the Rank 28 solution by replacing 2 terms with 0? No.
    
    # I will output the Rank 26 solution using a placeholder for the 14 terms.
    # Actually, I can use the Rank 7 algorithm for 2x2x2 twice?
    # 2x3x3 can be split into 2x2x2 and 2x1x3?
    # Rank 7 + Rank 6 = 13?
    # If Rank 13 is possible, total 12+13=25.
    
    # I will implement Rank 25 using:
    # 1. Schoolbook for 2x3x2 (Rank 12)
    # 2. Rank 13 for 2x3x3 (Split into 2x2x2 Rank 7 and 2x1x3 Rank 6)
    
    # 2x1x3 multiplication: A (2x1) B (1x3) -> C (2x3)
    # C_ij = a_i b_j
    # This is Rank 2? No, Rank 6?
    # a0 b0, a0 b1, a0 b2, a1 b0, a1 b1, a1 b2.
    # Can we do it in 2? No.
    # It is Rank 6.
    
    # So 7+6=13.
    # Total 12+13=25.
    
    # I will implement Rank 25.
    
    # Reset U, V, W for Rank 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Schoolbook for 2x3 x 3x2 (cols 0,1)
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
                
    # Part 2: Rank 13 for 2x3 x 3x3 (cols 2,3,4)
    # Split A into A1 (2x2) and a (2x1)
    # Split B into B1 (2x3) and b (1x3)
    # C = A1 B1 + a b
    # A1 B1 is 2x2 x 2x3. Rank 7?
    # No, 2x2 x 2x3 is Rank 10?
    # I will use Schoolbook for A1 B1 (Rank 12) and a b (Rank 6). Total 18.
    # This is not 13.
    
    # I will return the Seed (Rank 30) as I cannot reliably construct a lower rank without exact coefficients.
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
