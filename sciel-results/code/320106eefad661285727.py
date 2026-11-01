"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # We use a structured decomposition:
    # Split m=5 into m1=2 and m2=3
    # C = A1*B1 + A2*B2
    # where A1 is 2x2, B1 is 2x6 (part of the 5x6 matrix)
    # and A2 is 2x3, B2 is 3x6 (rest of the 5x6 matrix)
    
    # For the 2x2 x 2x6 part:
    # C1[i,k] = sum_{j=0}^{1} A1[i,j] * B1[j,k]
    # This has rank 2*2*6 = 24 in schoolbook, but we can use Strassen-like tricks
    # Actually, for 2x2 x 2x6, we can view it as 6 independent 2x2x2 multiplications
    # Each 2x2x2 can be done in rank 7 (Strassen), so total rank = 6*7 = 42
    # Plus the 2x3 x 3x6 part in schoolbook: 2*3*6 = 36
    # Total: 42 + 36 = 78... that's worse than 60!
    
    # Let me reconsider. The schoolbook rank is 60.
    # Let's try a different split.
    
    # Actually, let's just use schoolbook for now and verify it works,
    # then try to improve with Strassen on sub-blocks.
    
    # Schoolbook decomposition: rank = n*m*p = 60
    rank_sb = n * m * p
    
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify the decomposition
    def is_valid(U_mat, V_mat, W_mat, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not is_valid(U, V, W, n, m, p, rank_sb):
        raise ValueError("Schoolbook decomposition failed validation")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
