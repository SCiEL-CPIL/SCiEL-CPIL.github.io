"""Rank-55 tensor decomposition for 2x3 @ 3x10 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 10]
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    # Dimensions: A is 2x3, B is 3x10, C is 2x10
    n, m, p = 2, 3, 10
    
    # Schoolbook decomposition: Rank = n * m * p = 60
    rank = n * m * p
    U_mat = [[0] * rank for _ in range(n * m)]  # 6 x 60
    V_mat = [[0] * rank for _ in range(m * p)]  # 30 x 60
    W_mat = [[0] * rank for _ in range(p * n)]  # 20 x 60
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U_mat[i * m + j][idx] = 1
                V_mat[j * p + k][idx] = 1
                W_mat[k * n + i][idx] = 1
                idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
