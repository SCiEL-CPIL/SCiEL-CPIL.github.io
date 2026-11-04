import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # For 4x5x5 matrix multiplication
    # We use a block decomposition approach
    # Split A (4x5) into blocks, B (5x5) into blocks
    
    # Let's use a simple but effective approach:
    # We can decompose this as:
    # A = [A1 | A2] where A1 is 4x2, A2 is 4x3
    # B = [B1; B2] where B1 is 2x5, B2 is 3x5
    # C = A1*B1 + A2*B2
    
    # Rank of 4x2x5: at most 4*2*5 = 40, but can be lower
    # Rank of 4x3x5: at most 4*3*5 = 60, but can be lower
    
    # Actually, let's just use schoolbook for simplicity and correctness
    # Total rank: 4*5*5 = 100
    
    # But we can do better by using Strassen for sub-blocks
    
    # Let's try a different approach: use the known fact that
    # R(4,5,5) can be computed with rank less than 100
    
    # For now, let's implement a clean schoolbook decomposition
    # and verify it works correctly
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
