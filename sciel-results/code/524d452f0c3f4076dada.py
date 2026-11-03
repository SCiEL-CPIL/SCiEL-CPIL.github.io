import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 3x3 matrix multiplication.
    Rank = 27.
    """
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5s safety margin
    
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
                # Check time
                if time.time() > deadline:
                    # Return whatever we have? No, must be complete.
                    # If we timeout here, we should return the best valid result.
                    # Since we are in the middle of construction, we can't return a partial valid decomposition.
                    # But the loop is tiny (27 iterations). It won't timeout.
                    pass
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
