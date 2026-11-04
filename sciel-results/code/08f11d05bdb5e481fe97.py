import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    n, m, p = 4, 5, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(p):
            if time.time() > deadline:
                break
            for k in range(m):
                if time.time() > deadline:
                    break
                # U index: i * m + k
                # V index: k * p + j
                # W index: j * n + i
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
                r += 1
                
    # If we didn't fill all, we need to ensure the returned matrices are consistent
    # But since it's a direct loop, it will finish instantly.
    return {'U': U, 'V': V, 'W': W}
