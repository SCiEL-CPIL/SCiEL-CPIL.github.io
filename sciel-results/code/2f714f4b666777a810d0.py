import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target rank to beat: 32 (schoolbook)
    # Try to find rank 28 or 24
    best_rank = 32
    best_decomp = None
    
    # If we find a better one, update
    # Try ranks from 16 to 31
    for rank in range(16, 32):
        if time.time() > deadline - 5:
            break
            
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(nm)])
            V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(mp)])
            W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(pn)])
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_decomp = (U, V, W)
                break
    
    if best_decomp is None:
        rank = 32
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        U, V, W = best_decomp
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
