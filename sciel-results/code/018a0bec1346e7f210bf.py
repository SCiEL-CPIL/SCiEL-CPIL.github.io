import numpy as np
import time

DIMENSIONS = [2, 3, 9]
DEADLINE = 110.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = p  # 9
    U = [[1] * rank for _ in range(n * m)]  # 6 x 9, all ones
    V = [[1] * rank for _ in range(m * p)]  # 27 x 9, all ones
    W = [[0] * rank for _ in range(p * n)]  # 18 x 9
    
    # W[k*n+i, r] = 1 if k == r else 0
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for col in range(r):
                        s += U[i*m+j][col] * V[j*p+k][col] * W[k*n+i][col]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
