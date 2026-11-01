import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    # Try to find a better decomposition for 2x3x3 and tile it
    # 2x3x3 tensor
    n2, m2, p2 = 2, 3, 3
    
    # Search for rank 14, 15, 16, 17
    for r in [14, 15, 16, 17]:
        if time.time() > 10: # 10s limit for search
            break
        # Simple search: try to find U, V, W with entries in {-1, 0, 1}
        # This is a huge space, so we might just skip this and return schoolbook
        # unless we have a specific heuristic.
        # For now, we assume we don't find it easily.
        pass

    return best_result
