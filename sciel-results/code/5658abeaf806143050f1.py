"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 48
    # We'll try to find a lower rank decomposition
    # For 3x4 * 4x4, the bilinear complexity is at most 48 (schoolbook)
    # but may be lower. Let's try a few known lower bounds.
    
    # The tensor T is of shape (n*m, m*p, p*n) = (12, 16, 12)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We'll try ranks from 32 to 47 and see if we can find a valid decomposition
    # using integer or half-integer coefficients.
    
    # For now, let's return the schoolbook decomposition which is guaranteed valid.
    # A more sophisticated search would try to reduce the rank.
    
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
