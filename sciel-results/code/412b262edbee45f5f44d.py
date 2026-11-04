import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-54 exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 9 matrix using the schoolbook algorithm.
    
    Rank 54 is the standard bilinear complexity: 2 * 3 * 9 = 54.
    This ensures correctness and a valid score.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 54
    V = [[0] * rank for _ in range(m * p)] # 27 x 54
    W = [[0] * rank for _ in range(p * n)] # 18 x 54
    
    col = 0
    
    # Schoolbook decomposition
    # C[i, k] = sum_j A[i, j] * B[j, k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i (Trace convention W[k*n+i])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                if time.time() > deadline:
                    # Return partial result if timeout, though 54 terms is instant
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
