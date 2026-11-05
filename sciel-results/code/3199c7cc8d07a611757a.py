import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 3, 3, 7
    
    rank1 = 42  # Part 1: 3x2 * 2x7 schoolbook
    rank2 = 3   # Part 2: 3x1 * 1x7 (rank-1 per row)
    total_rank = rank1 + rank2  # 45
    
    U = np.zeros((9, total_rank), dtype=int)
    V = np.zeros((21, total_rank), dtype=int)
    W = np.zeros((21, total_rank), dtype=int)
    
    # Part 1: columns 0..41
    col = 0
    for i in range(3):      # row of A/C
        for j in range(2):  # col of A / row of B
            for k in range(7):  # col of B/C
                U[i*3 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
    
    # Part 2: columns 42..44
    for i in range(3):
        U[i*3 + 2, col] = 1
        for k in range(7):
            V[14 + k, col] = 1
            W[k*3 + i, col] = 1
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
