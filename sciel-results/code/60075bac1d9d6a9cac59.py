import numpy as np
import random
import time

def construct():
    n, m, p = 2, 3, 3
    target_rank = 15
    
    def check(U, V, W, rank):
        # U: (6, rank), V: (9, rank), W: (6, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        return False
        return True
    
    # Try rank 15
    deadline = time.time() + 120
    
    # Randomized search with integer coefficients
    best_result = None
    
    for attempt in range(100000):
        if time.time() > deadline - 5:
            break
        
        # Generate random matrices with small integer entries
        entries = [-1, 0, 0, 0, 1]  # sparse
        U = np.random.choice(entries, size=(6, target_rank))
        V = np.random.choice(entries, size=(9, target_rank))
        W = np.random.choice(entries, size=(6, target_rank))
        
        if check(U, V, W, target_rank):
            best_result = (U, V, W, target_rank)
            break
    
    if best_result is None:
        # Fallback to schoolbook rank 18
        rank = 18
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(6)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
