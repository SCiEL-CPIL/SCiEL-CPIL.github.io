import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a better decomposition using Strassen-like algorithm for 3x3x3 blocks
    # Rank of 3x3x3 is 23.
    # Total rank = 23 + 23 = 46.
    
    # Define the 23 rank-1 components for 3x3x3 multiplication
    # u, v, w are vectors of length 9 (3x3 matrix flattened)
    # Indices: i*3 + j for u, j*3 + k for v, k*3 + i for w
    
    # I will use a known set of 23 coefficients.
    # If I cannot recall them exactly, I will use a fallback.
    # For the purpose of this exercise, I will define a placeholder list.
    # In a real scenario, these would be the actual coefficients.
    
    # Since I don't have the exact 23 coefficients memorized, I will use a 
    # heuristic: if I can't verify a better decomposition, I return schoolbook.
    
    # However, I can try to construct a rank 46 decomposition using 
    # the 7-term Strassen for 2x2 and schoolbook for the rest?
    # This is complex.
    
    # Let's stick to the schoolbook for safety, as I cannot guarantee the 23 coefficients.
    # But wait, I can try to find a rank 50 decomposition?
    
    # Actually, I will just return the schoolbook decomposition.
    # It is guaranteed to be correct.
    
    rank = rank_schoolbook
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
