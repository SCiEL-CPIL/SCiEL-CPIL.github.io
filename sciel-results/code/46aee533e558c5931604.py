import time
import sys

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = 120  # 120 seconds limit, keep safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the seed program interface
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
    if time.time() - start_time > deadline - 10:
         # Return current best (which is the schoolbook)
         pass
         
    return {"U": U, "V": V, "W": W}
