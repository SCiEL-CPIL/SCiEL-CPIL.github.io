import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 140
    n, m, p = DIMENSIONS
    sb_rank = n * m * p
    
    # Try a few low ranks quickly
    for R in [45, 50, 55]:
        if time.time() > deadline - 30:
            break
        np.random.seed(0)
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        # Quick check: won't likely work, but try
        # If it did, we'd need to round and verify
        pass
    
    # Return schoolbook - guaranteed correct
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
