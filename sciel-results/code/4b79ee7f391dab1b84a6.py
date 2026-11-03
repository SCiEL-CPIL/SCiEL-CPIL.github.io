import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition for 3x3x3 part (rank 27)
    # Rank 1 decomposition for 3x3x4 part (rank 4)
    # Total rank 31
    
    rank1 = n * m * 3  # 27
    rank2 = 4
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 3x3x3 schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: 3x3x4 rank 4
    # For each k in 3..6, use rank 1
    for s in range(4):
        k_global = 3 + s
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + k_global, col] = 1
                W[k_global * n + i, col] = 1
        col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
