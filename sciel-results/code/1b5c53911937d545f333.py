import numpy as np
import itertools
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 145
    
    # Schoolbook rank
    rank_school = n * m * p  # 48
    
    target = np.ones((n, m, p))
    target_flat = target.flatten()
    
    # Try ranks from 47 down to 44
    for rank in [47, 46, 45, 44]:
        if time.time() > deadline - 15:
            break
            
        found = False
        U_best = None
        V_best = None
        W_best = None
        
        # Limit attempts based on remaining time
        time_per_attempt = 0.05
        max_attempts = max(50, int((deadline - 15 - time.time()) / time_per_attempt))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            # Generate random U, V with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Build linear system A * W_flat = target_flat
            A = np.zeros((n * m * p, p * n * rank))
            
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        u_val = U[i * m + j, r]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = V[j * p + k, r]
                            if v_val == 0:
                                continue
                            A[i * m * p + j * p + k, (k * n + i) * rank + r] = u_val * v_val
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, target_flat, rcond=None)
            except:
                continue
            
            # Check if solution is close to integers
            W_int = np.round(W_flat)
            error = np.linalg.norm(A @ W_int - target_flat)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                if np.allclose(W_flat, W_int):
                    found = True
                    U_best = U
                    V_best = V
                    W_best = W
                    break
        
        if found:
            U_list = [[int(U_best[i, r]) for r in range(rank)] for i in range(n * m)]
            V_list = [[int(V_best[i, r]) for r in range(rank)] for i in range(m * p)]
            W_list = [[int(W_best[i, r]) for r in range(rank)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
