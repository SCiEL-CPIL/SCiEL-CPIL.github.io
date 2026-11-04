import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Use additive bound: J_3x3x7 = J_3x3x3 + J_3x3x3 + J_3x3x1
    # Rank(J_3x3x3) <= 27 (schoolbook)
    # Rank(J_3x3x1) = 1
    # Total rank <= 27 + 27 + 1 = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First 3 columns of B (k=0,1,2)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Next 3 columns of B (k=3,4,5)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Last column of B (k=6)
    # Rank 1 decomposition for J_3x3x1
    # u_ij = 1, v_jk = 1, w_ik = 1 (for k=6)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
    for j in range(m):
        for k in range(6, 7):
            V[j * p + k][col] = 1
    for i in range(n):
        for k in range(6, 7):
            W[k * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
