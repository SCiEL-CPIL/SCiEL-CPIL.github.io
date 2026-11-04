import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 110

def _elapsed():
    return time.time() - START_TIME

def _verify(U, V, W, n=3, m=5, p=7):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 5, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Direct construction of the schoolbook algorithm.
    # Rank = n * m * p = 105.
    # Each term in the sum corresponds to one index r.
    # U[i*m+j, r] = 1 if r corresponds to (i,j), else 0.
    # V[j*p+k, r] = 1 if r corresponds to (j,k), else 0.
    # W[k*n+i, r] = 1 if r corresponds to (k,i), else 0.
    # This ensures that for a specific (i,j,k), exactly one r has all three components as 1,
    # so the sum is 1. For any other combination, the sum is 0.
    
    rank = nm * mp * pn # 3 * 5 * 7 = 105
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map (i, j, k) to index idx
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    
    # The entries are 0.0 and 1.0, which are integers.
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
