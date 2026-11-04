import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Deadline guard
    start_time = time.time()
    deadline = 110  # 10 seconds safety margin
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_list = [[0] * schoolbook_rank for _ in range(n * m)]
    V_list = [[0] * schoolbook_rank for _ in range(m * p)]
    W_list = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][column] = 1
                V_list[j * p + k][column] = 1
                W_list[k * n + i][column] = 1
                column += 1
    
    # Return schoolbook if time is running out
    if time.time() > deadline:
        return {"U": U_list, "V": V_list, "W": W_list}
        
    # Attempt to find a better decomposition
    # For now, return schoolbook as a safe baseline
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
