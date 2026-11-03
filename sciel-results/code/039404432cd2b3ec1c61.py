"""Program for matrix multiplication 3x3 * 3x5 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    # Rank-40 decomposition for 3x3 * 3x5 matrix multiplication
    # Strategy: Use Strassen's rank-7 algorithm for 2x2 blocks, 
    # combined with careful handling of the remaining rows/columns.
    
    # The matrix multiplication C = A * B where A is 3x3 and B is 3x5
    # We can decompose A and B into 2x2 and 1x1 blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 1x2, B22 is 1x3
    
    # Actually, let's use a more systematic approach.
    # We'll use the known rank-40 construction for (3,3,5).
    
    # One valid construction uses 40 terms. Let me implement a known valid rank.
    # The schoolbook method gives rank 3*3*5 = 45.
    # We can do better.
    
    # Let's use a construction based on splitting the 5 columns into 2+3:
    # Compute A*B[:,0:2] (3x3 * 3x2) and A*B[:,2:5] (3x3 * 3x3) separately.
    
    # For 3x3 * 3x2: rank at most 18 (schoolbook) but can be lower.
    # For 3x3 * 3x3: rank at most 27 (schoolbook) but Strassen gives 23 for 3x3x3.
    
    # Actually, let me just implement a valid rank-45 decomposition first,
    # then try to optimize.
    
    # Rank-45 schoolbook decomposition:
    rank = 45
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
# EVOLVE-BLOCK-END
