"""Matrix multiplication 3,3,5 rank minimization."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x5.
    Currently returns the schoolbook decomposition of rank 45.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook: rank 45
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
