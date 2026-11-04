import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return best found so far (which is the full schoolbook if we haven't finished)
                    # Since we are filling it sequentially, we might have an incomplete tensor.
                    # It's safer to just return the full schoolbook if we can't finish the loop, 
                    # but the loop is fast (60 iterations).
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
        
    return {"U": U, "V": V, "W": W}
