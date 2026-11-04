import time

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Constructs the schoolbook tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Rank = 3 * 3 * 5 = 45.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time just in case, though this loop is very fast
                if time.time() > deadline:
                    # If we run out of time (highly unlikely), return what we have so far
                    # But since we are building it sequentially, an incomplete matrix is invalid.
                    # Better to just finish, it's < 1ms.
                    pass
                    
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
