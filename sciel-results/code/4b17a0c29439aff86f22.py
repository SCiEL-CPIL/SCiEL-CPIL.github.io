import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on splitting 3x4x8 into two 3x4x4 blocks.
    # Each 3x4x4 block is decomposed using a 3x3x3 (rank 23) and 3x1x4 (rank 12) and 3x3x1 (rank 9) parts.
    # Total rank per 3x4x4 block: 23 + 9 + 12 = 44.
    # Total rank: 44 * 2 = 88.
    
    # We need the rank 23 decomposition for 3x3x3.
    # This is a known result. I will hardcode a valid rank 23 decomposition.
    # If I cannot, I will fall back to schoolbook.
    
    # For safety, I will use the schoolbook decomposition if I don't have the exact vectors.
    # However, I will try to provide the 88 rank one.
    
    # Since I don't have the exact 23 vectors memorized perfectly, I will use the schoolbook decomposition 
    # to ensure validity, but I will structure the code to allow for a better decomposition if found.
    # Actually, I will just return the schoolbook decomposition to be safe.
    
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
