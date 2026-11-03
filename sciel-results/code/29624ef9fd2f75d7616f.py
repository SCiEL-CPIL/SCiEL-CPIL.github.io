import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Matrix multiplication tensor decomposition for 3x4 * 4x7 -> 3x7.
    
    We use a block (slice-and-dice) strategy:
    - Partition rows of A (3x4) into blocks of size [2, 1]
    - Partition columns of A into blocks of size [2, 2]
    
    This gives 4 sub-problems:
    1. A[0:2, 0:2] * B[0:2, 0:7] -> C[0:2, 0:7]  (2x2 x 7)
    2. A[0:2, 2:4] * B[2:4, 0:7] -> C[0:2, 0:7]  (2x2 x 7)
    3. A[2:3, 0:2] * B[0:2, 0:7] -> C[2:3, 0:7]  (1x2 x 7)
    4. A[2:3, 2:4] * B[2:4, 0:7] -> C[2:3, 0:7]  (1x2 x 7)
    
    For each sub-problem, we use the schoolbook (rank = n_sub * m_sub * p_sub) decomposition.
    Total rank = 2*2*7 + 2*2*7 + 1*2*7 + 1*2*7 = 28 + 28 + 14 + 14 = 84
    
    This matches the global schoolbook rank, which is the trivial upper bound.
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # Start with schoolbook rank as fallback
    rank = n * m * p  # 84
    
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
