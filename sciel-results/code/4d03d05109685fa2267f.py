import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication M_{3,4,5}.
    Uses the schoolbook (standard) decomposition which provides a valid
    decomposition of rank 60. Returns a dictionary with keys 'U', 'V', 'W'
    containing the factor matrices as lists of lists.
    """
    n, m, p = DIMENSIONS
    total_rank = n * m * p
    
    U_list = [[0] * total_rank for _ in range(n * m)]
    V_list = [[0] * total_rank for _ in range(m * p)]
    W_list = [[0] * total_rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][r] = 1
                V_list[j * p + k][r] = 1
                W_list[k * n + i][r] = 1
                r += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
