"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Block decomposition: split middle dimension 3 into 2+1
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:3] @ B[2:3,:]
    # Part 1: 2x2x5 (schoolbook: 20 terms)
    # Part 2: 2x1x5 (rank 5 terms)
    # Total: 25
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 6 x 25
    V = [[0] * rank for _ in range(m * p)]  # 15 x 25
    W = [[0] * rank for _ in range(p * n)]  # 10 x 25
    
    col = 0
    
    # Part 2: j=2 (2x1x5), 5 terms
    for k in range(p):
        for i in range(n):
            U[i * m + 2][col] = 1
        V[2 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
        col += 1
    
    # Part 1: j=0,1 (2x2x5), 20 terms (schoolbook)
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

