import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: ensure we finish well before 120s
    start_time = time.time()
    deadline = 115.0  # 5 seconds margin
    
    n, m, p = DIMENSIONS
    # The schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Using lists of lists as per interface, but numpy is available for speed if needed.
    # Given the small size, lists are fine.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check deadline (unlikely to be hit, but required by directive)
                if time.time() > deadline:
                    # Return current best (which is the full decomposition if we finished, 
                    # or a partial one if we didn't, but partial is invalid. 
                    # So we just break and hope it's done. It is O(36) ops, so it's instant.)
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
