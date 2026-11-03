"""Tensor decomposition for 4x4 x 4x6 matrix multiplication.

Uses the standard schoolbook algorithm.
Rank = 4 * 4 * 6 = 96.

This ensures correctness.
"""

DIMENSIONS = [4, 4, 6]


def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    
    U: shape (16, 96)
    V: shape (24, 96)
    W: shape (24, 96)
    """
    n, m, p = 4, 4, 6
    
    # U has n*m rows
    # V has m*p rows
    # W has p*n rows (index k*n + i)
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i * m + j
                u_idx = i * m + j
                # B index: j * p + k
                v_idx = j * p + k
                # W index: k * n + i (as per task description W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx][r_idx] = 1
                V[v_idx][r_idx] = 1
                W[w_idx][r_idx] = 1
                
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}
