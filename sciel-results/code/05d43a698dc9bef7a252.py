import numpy as np
import random
import time

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 140
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Quick randomized search for lower rank (limited attempts)
    for target_rank in [rank - 1, rank - 2]:
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Random perturbation of schoolbook
            U_try = U.astype(float).copy()
            V_try = V.astype(float).copy()
            W_try = W.astype(float).copy()
            
            # Add small random noise to some entries
            for r in range(min(target_rank, rank)):
                if random.random() < 0.3:
                    U_try[:, r] += random.gauss(0, 0.1)
                if random.random() < 0.3:
                    V_try[:, r] += random.gauss(0, 0.1)
                if random.random() < 0.3:
                    W_try[:, r] += random.gauss(0, 0.1)
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_try[i*m+j, :] * V_try[j*p+k, :] * W_try[k*n+i, :])
                        if abs(val - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
