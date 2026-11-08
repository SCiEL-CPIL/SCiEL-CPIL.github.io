import numpy as np
import time

DEADLINE_SECONDS = 115.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # 2x2 Strassen factor matrices
    # U2, V2, W2 are 4x7 matrices
    # Row indices: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Build the rank-49 decomposition using Kronecker product structure
    # For each (alpha, beta) in 0..6 x 0..6:
    #   r = alpha * 7 + beta
    #   U[i*m+j, r] = U2[i1*2+j1, alpha] * U2[i2*2+j2, beta]
    #   V[j*p+k, r] = V2[j1*2+k1, alpha] * V2[j2*2+k2, beta]
    #   W[k*n+i, r] = W2[k1*2+i1, alpha] * W2[k2*2+i2, beta]
    
    for alpha in range(7):
        for beta in range(7):
            r = alpha * 7 + beta
            
            # Fill U
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for j in range(4):
                    j1 = j // 2
                    j2 = j % 2
                    val = U2[i1 * 2 + j1, alpha] * U2[i2 * 2 + j2, beta]
                    if val != 0:
                        U[i * m + j, r] = val
            
            # Fill V
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                for k in range(4):
                    k1 = k // 2
                    k2 = k % 2
                    val = V2[j1 * 2 + k1, alpha] * V2[j2 * 2 + k2, beta]
                    if val != 0:
                        V[j * p + k, r] = val
            
            # Fill W
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                for i in range(4):
                    i1 = i // 2
                    i2 = i % 2
                    val = W2[k1 * 2 + i1, alpha] * W2[k2 * 2 + i2, beta]
                    if val != 0:
                        W[k * n + i, r] = val
    
    # Verification: check that T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # This is a sanity check. If it fails, the decomposition is incorrect.
    # We do a spot check on a few entries to be safe.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    # Decomposition is incorrect; raise to signal failure
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {s}")
    
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
