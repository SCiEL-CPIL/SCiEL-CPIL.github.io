"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Try block decomposition: split m=5 into 3+2
    # C = A[:,0:3] * B[0:3,:] + A[:,3:5] * B[3:5,:]
    # Block 1: 3x3x6 (two 3x3x3)
    # Block 2: 3x2x6 (two 3x2x3)
    
    # Use schoolbook for now, but with explicit deadline guard
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we ran out of time, return what we have (may be incomplete)
    # But for schoolbook, it's fast enough
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
