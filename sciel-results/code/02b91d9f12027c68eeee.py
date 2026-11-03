"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 110  # safety margin before 120s limit

    # For 2x3 * 3x6 matrix multiplication, the schoolbook method gives rank 36.
    # We can exploit the rectangular structure: the 2x3 input matrix A can be
    # partitioned into two 1x3 row vectors. Each row vector multiplies the 3x6
    # matrix B to produce a 1x6 row of C. These two sub-problems are independent.
    #
    # For a 1x3 * 3x6 multiplication, the tensor rank is 3*6 = 18 (schoolbook).
    # So total rank = 2 * 18 = 36.
    #
    # However, let's try to find a better decomposition. The key insight is that
    # matrix multiplication is bilinear, and the tensor T_{i,j,k} = 1 for all valid
    # indices. The rank of this tensor is at most n*m*p = 36, but could be lower.
    #
    # For now, use the schoolbook construction which is known to be valid.
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
# EVOLVE-BLOCK-END
