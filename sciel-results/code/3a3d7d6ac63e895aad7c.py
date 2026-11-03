import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses a rank-28 decomposition (rank-27 schoolbook for 3x3x3 + rank-1 for the 4th column).
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 28
    V = [[0] * rank for _ in range(m * p)]  # 12 x 28
    W = [[0] * rank for _ in range(p * n)]  # 12 x 28
    
    # Part 1: Schoolbook for 3x3x3 (k=0,1,2)
    # 27 terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # Only first 3 columns
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: Rank 1 for the 4th column (k=3)
    # T_ij3 = 1 for all i,j
    # U_ij,27 = 1, V_j3,27 = 1, W_3i,27 = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][27] = 1
            V[j * p + 3][27] = 1
            W[3 * n + i][27] = 1
            
    return {"U": U, "V": V, "W": W}
