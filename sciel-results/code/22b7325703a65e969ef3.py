"""Program to find a low-rank exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    rank = n * m * p  # schoolbook rank = 36
    
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For 2x3 x 3x6, we can split the 6 into two 3s.
    # So we do two independent 2x3 x 3x3 multiplications.
    # Each 2x3 x 3x3 has schoolbook rank 18.
    # Total rank = 36.
    # 
    # Actually, let's just do the schoolbook for the full thing.
    # It's the safest bet.
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
