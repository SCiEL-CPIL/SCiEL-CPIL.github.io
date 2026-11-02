import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 3, 3, 6
    
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Create the schoolbook decomposition (rank 54)
    U, V, W, rank = make_schoolbook(n, m, p)
    
    # Verify and return
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
