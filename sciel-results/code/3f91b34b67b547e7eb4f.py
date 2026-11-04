<<<<<<< SEARCH
"""Direct construction of the exact tensor decomposition for (2,3,5) matrix multiplication."""

def construct():
    n, m, p = 2, 3, 5
    rank = n * m * p  # 30; known minimal for (2,3,5) as R(2,3,5) = R(1,3,5) + R(1,3,5) = 15+15
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 30
    V = [[0] * rank for _ in range(m * p)]   # 15 x 30
    W = [[0] * rank for _ in range(p * n)]   # 10 x 30
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
=======
"""Direct construction of the exact tensor decomposition for (2,3,5) matrix multiplication."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 5
    rank = n * m * p  # 30; known minimal for (2,3,5) as R(2,3,5) = R(1,3,5) + R(1,3,5) = 15+15
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 30
    V = [[0] * rank for _ in range(m * p)]   # 15 x 30
    W = [[0] * rank for _ in range(p * n)]   # 10 x 30
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        if time.time() > deadline:
            break
    if time.time() > deadline:
        break
    
    # If the loop was interrupted, we might have an incomplete decomposition.
    # However, the construction is deterministic and fast, so this is unlikely.
    # If incomplete, we should probably return the full valid one or handle the error.
    # Since the loop is very fast (30 iterations), it will always complete.
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
