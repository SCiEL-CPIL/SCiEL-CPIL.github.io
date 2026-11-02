# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # We use a block decomposition.
    # Split B into 4 blocks of 2x2.
    # A is 3x4, split into A0 (3x2) and A1 (3x2).
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # Each term is M_{3,2,2}.
    # M_{3,2,2} is 3 independent M_{1,2,2}.
    # Rank of M_{1,2,2} is 3.
    # Total rank = 4 * 3 * 3 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define M_{1,2,2} rank 3 decomposition
    # Inputs: a0, a1 (indices 0, 1 in A-block)
    #         b00, b01, b10, b11 (indices in B-block)
    # Outputs: c0, c1
    # 
    # Decomposition:
    # 1. m1 = (a0 + a1) * (b00 + b10)
    # 2. m2 = a0 * (b00 + b10)
    # 3. m3 = a1 * (b10 + b11)
    # This was found to be incorrect.
    
    # Let's use a known valid Rank 3 decomposition for M_{1,2,2} if available.
    # If not, use Rank 4.
    # For now, I will use Rank 4 to ensure correctness, giving Rank 48.
    # But I want to improve.
    
    # I will implement Rank 36 with a placeholder that I will verify.
    # Since I cannot verify interactively, I will use a safe Rank 4 decomposition for M_{1,2,2}.
    # This results in Rank 48.
    
    # To improve, I will use the trivial Rank 48.
    
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
