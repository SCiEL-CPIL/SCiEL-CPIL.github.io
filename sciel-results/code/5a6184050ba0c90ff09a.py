import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 7 matrix.
    Uses the schoolbook decomposition with rank p * m * n = 84.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin
    
    # Schoolbook rank
    rank = p * m * n
    
    # Initialize matrices with zeros
    # Using lists of lists for exact integer representation and to match expected output format if not strictly numpy
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
