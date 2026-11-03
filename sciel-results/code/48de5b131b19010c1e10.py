import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Standard schoolbook multiplication rank is n*m*p
    rank = n * m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for schoolbook multiplication
    # The tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # For schoolbook, each term in the sum corresponds to a unique (i,j,k) tuple.
    # We map (i,j,k) to a rank index r.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m + j) * p + k
                if r >= rank:
                    # Safety check, shouldn't happen with rank = n*m*p
                    continue
                
                # U index: (i, j) -> row i*m + j
                u_idx = i * m + j
                # V index: (j, k) -> row j*p + k
                v_idx = j * p + k
                # W index: (k, i) -> row k*n + i (following W[k*n+i, r] convention)
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
                if time.time() > deadline:
                    # If we run out of time, we return what we have so far.
                    # However, schoolbook is O(nmp) which is 84 iterations, very fast.
                    # This timeout is unlikely to trigger.
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
