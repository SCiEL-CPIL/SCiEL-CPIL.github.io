# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook rank 72
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Try Rank 68
    # 23 terms for 3x3x3
    # 5 * 9 = 45 terms for 3x3x5
    # Total 68
    
    # Known rank-23 decomposition for 3x3x3
    # Terms (u, v, w) as sparse dicts or dense vectors
    # I will use the decomposition from the GAP package.
    # If not available, I use a fallback.
    
    # Fallback: Return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
