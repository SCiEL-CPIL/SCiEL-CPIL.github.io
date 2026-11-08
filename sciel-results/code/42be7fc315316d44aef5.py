import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for 
    multiplying a 3x4 matrix by a 4x5 matrix.
    
    Currently implements the schoolbook algorithm with rank 60.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
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
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    return {"U": U, "V": V, "W": W}
