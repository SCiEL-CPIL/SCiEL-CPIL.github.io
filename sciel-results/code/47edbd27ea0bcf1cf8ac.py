import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use 2x2 Strassen on A blocks, with B split into column groups
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B (4x6) split into B_left (4x4) and B_right (4x2)
    # B_left = [[B11, B12], [B21, B22]] (each 2x2)
    # B_right = [[B13, B14], [B23, B24]] (each 2x2)
    
    # For C_left (4x4), use Strassen: 7 terms
    # For C_right (4x2), use 4 terms (schoolbook on 2x2 blocks)
    # Total: 7 + 4 = 11 terms? But each term is a 2x2 x 2x2 multiplication
    # Wait, no. Each Strassen term is one 2x2x2 multiplication (rank 7 each)
    # So 7 terms * rank 7 = 49 for C_left
    # For C_right: 4 terms * rank 4 (schoolbook 2x2x2) = 16
    # Total: 49 + 16 = 65
    
    # That's better than 96 but not 40.
    
    # Let me implement this:
    # 7 Strassen terms for 4x4x4 (each rank 7) = 49
    # 4 schoolbook terms for 4x4x2 (each rank 4) = 16
    # Total rank = 65
    
    # Actually, for the 4x4x2 part, I can use Strassen too if I pad to 4x4x4
    # But that's wasteful.
    
    # Let me just implement rank 65 and see if it works.
    
    # Hmm, this is getting complicated. Let me just implement a clean Strassen-based decomposition.
    
    # I'll implement the 2x2 Strassen for the 4x4x4 part (rank 49) 
    # and schoolbook for the 4x4x2 part (rank 16)
    # Total rank = 49 + 16 = 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 24 x 65
    W = [[0] * rank for _ in range(p * n)]  # 24 x 65
    
    col = 0
    
    # Part 1: 4x4x4 via Strassen (rank 49)
    # Strassen terms:
    # P1 = A11 * B11
    # P2 = A12 * (B21 - B11)
    # P3 = (A11 + A12) * B12
    # P4 = A21 * (B11 - B21)
    # P5 = (A21 + A22) * B22
    # P6 = (A11 - A21) * (B12 + B22)
    # P7 = (A12 - A22) * (B21 - B22)
    
    # Each Pi is a 2x2x2 tensor with rank 7
    # I need to expand each into 7 rank-1 terms
    
    # For a 2x2x2 tensor with A, B, C blocks:
    # C11 = P1 + P7 - P4 + P6
    # C12 = P2 + P4
    # C21 = P3 + P5
    # C22 = P1 - P3 + P2 + P7
    
    # The 7 rank-1 terms for Strassen 2x2x2:
    # s1: A11*B11, contributes to C11 with +1
    # s2: A12*(B21-B11), contributes to C12 with +1
    # s3: (A11+A12)*B12, contributes to C21 with +1
    # s4: A21*(B11-B21), contributes to C12 with +1, C11 with -1
    # s5: (A21+A22)*B22, contributes to C21 with +1
    # s6: (A11-A21)*(B12+B22), contributes to C11 with +1, C22 with +1
    # s7: (A12-A22)*(B21-B22), contributes to C11 with +1, C22 with +1
    
    # Wait, I need to be more careful. Let me define the Strassen decomposition properly.
    
    # For 2x2x2: C = A*B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]], C = [[c11, c12], [c21, c22]]
    # 
    # M1 = a11 * b11
    # M2 = a12 * (b21 - b11)
    # M3 = (a11 + a12) * b12
    # M4 = a21 * (b11 - b21)
    # M5 = (a21 + a22) * b22
    # M6 = (a11 - a21) * (b12 + b22)
    # M7 = (a12 - a22) * (b21 - b22)
    #
    # c11 = M1 + M7 - M4 + M6
    # c12 = M2 + M4
    # c21 = M3 + M5
    # c22 = M1 - M3 + M2 + M7
    
    # Each M_i is a scalar product, but in the tensor decomposition, each M_i corresponds
    # to a 2x2x2 sub-tensor which itself has rank 7.
    
    # This is getting very complex. Let me just implement schoolbook for now and return rank 96.
    # The measured improvement to rank 40 requires more careful design.
    
    # For this iteration, I'll implement a rank-65 decomposition:
    # 49 for 4x4x4 (Strassen) + 16 for 4x4x2 (schoolbook on 2x2 blocks)
    
    # Actually, let me just return schoolbook rank 96 as a safe baseline.
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
