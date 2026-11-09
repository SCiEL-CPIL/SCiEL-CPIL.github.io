import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # Return best result found so far (which is the full schoolbook if not interrupted)
                    # Since we are building it sequentially, if interrupted, we return what we have?
                    # No, we must return a valid decomposition.
                    # The schoolbook loop is very fast for 54 columns.
                    # It will finish in microseconds.
                    pass
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
