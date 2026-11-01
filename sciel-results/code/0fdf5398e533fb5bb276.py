import numpy as np
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication 2x3 by 3x7 can be viewed as 3 independent
    # multiplications of 2x1 by 1x7 matrices.
    # The rank of M_{2,1,7} is 2*7 = 14.
    # Total rank is 3 * 14 = 42.
    rank_total = m * n * p
    
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    col = 0
    for j in range(m):
        # Block j corresponds to column j of A and row j of B
        for i in range(n):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
                
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
