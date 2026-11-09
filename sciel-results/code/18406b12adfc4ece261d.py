import numpy as np
import random
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_np(U, V, W, rank):
        U_r = np.array(U).reshape(n, m, rank)
        V_r = np.array(V).reshape(m, p, rank)
        W_r = np.array(W).reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    best_result = None
    best_rank = n * m * p
    
    # Lower bound
    lb = max(n*m, m*p, p*n)
    
    for target_rank in range(lb, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Use numpy for fast random generation
            # Integers -1, 0, 1
            U = np.random.choice([-1, 0, 1], size=(nm, target_rank))
            V = np.random.choice([-1, 0, 1], size=(mp, target_rank))
            W = np.random.choice([-1, 0, 1], size=(pn, target_rank))
            
            if verify_np(U, V, W, target_rank):
                # Convert to list of lists
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
        
        if best_result is not None:
            break

    if best_result is not None:
        return best_result
    
    return schoolbook()

if __name__ == "__main__":
    res = construct()
    # ... verification code ...
