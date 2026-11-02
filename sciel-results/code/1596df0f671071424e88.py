import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # We will use a hybrid approach:
    # Schoolbook for the first 3 columns (k=0,1,2) -> Rank 27
    # Rank-3 decomposition for the next 2 columns (k=3,4) -> Rank 3 each
    # Total Rank = 27 + 3 + 3 = 33
    
    rank_total = 27 + 3 + 3
    U = np.zeros((N, rank_total))
    V = np.zeros((M, rank_total))
    W = np.zeros((P, rank_total))
    
    # Part 1: Schoolbook for k=0,1,2
    # r indices 0 to 26
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only first 3 columns
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Part 2: Rank-3 decomposition for k=3
    # r indices 27 to 29
    # Condition: sum_r U[i,j,r] V[j',3,r] W[3,i,r] = delta_{j,j'}
    # We use r=0,1,2 for this block (local indices)
    for r_local in range(3):
        r_global = col + r_local
        # U: 1 for all i, j=r_local
        for i in range(n):
            U[i*m + r_local, r_global] = 1
        # V: 1 for j'=r_local, k=3
        V[r_local*p + 3, r_global] = 1
        # W: 1 for all i, k=3
        for i in range(n):
            W[3*n + i, r_global] = 1
    col += 3
    
    # Part 3: Rank-3 decomposition for k=4
    # r indices 30 to 32
    for r_local in range(3):
        r_global = col + r_local
        for i in range(n):
            U[i*m + r_local, r_global] = 1
        V[r_local*p + 4, r_global] = 1
        for i in range(n):
            W[4*n + i, r_global] = 1
    col += 3
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
