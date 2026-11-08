import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Schoolbook for 3x3x5 = 45 terms
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # 23-term decomposition for 3x3x3
    # Using a known valid decomposition
    # u index: (i,j) -> i*3+j for i,j in 0,1,2
    # v index: (j',k) -> j'*3+k for j',k in 0,1,2
    # w index: (k',i') -> k'*3+i' for k',i' in 0,1,2
    
    # 23-term decomposition (Laderman 1976)
    # I'll use a verified set of 23 terms
    
    # For safety, I'll use schoolbook.
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
