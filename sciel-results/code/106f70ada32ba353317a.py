import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    R = n * m * p  # 84 for 3x4x7

    # Initialize factor matrices
    # U has shape (n*m, R)
    # V has shape (m*p, R)
    # W has shape (p*n, R)
    # Using lists of lists to match the return interface expectation if not strictly numpy
    # But the prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # and returns a dict with keys 'U', 'V', 'W'.
    # The current program returns .tolist().
    
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
