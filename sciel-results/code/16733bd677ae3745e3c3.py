import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s wall-clock kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_find_rank(R, time_budget=10.0):
        start_time = time.time()
        max_attempts = 5000
        values = [-1.0, -0.5, 0.5, 1.0]
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Generate random U and V with controlled sparsity
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # For each column r, pick a structured pattern
            for r in range(R):
                # Each column gets 1-3 non-zeros in U and V
                n_u_nonzero = random.randint(1, 3)
                n_v_nonzero = random.randint(1, 3)
                
                u_indices = random.sample(range(nm), min(n_u_nonzero, nm))
                v_indices = random.sample(range(mp), min(n_v_nonzero, mp))
                
                for idx in u_indices:
                    U[idx, r] = random.choice(values)
                for idx in v_indices:
                    V[idx, r] = random.choice(values)
            
            # Build coefficient matrix C for W
            # C[e, col] where e = i*m*p + j*p + k, col = r*pn + (k*n+i)
            C = np.zeros((n*m*p, pn*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i*m*p + j*p + k
                        for r in range(R):
                            u_val = U[i*m+j, r]
                            v_val = V[j*p+k, r]
                            w_idx = k*n + i
                            col = r * pn + w_idx
                            C[e, col] = u_val * v_val
            
            b = np.ones(n*m*p)
            
            # Solve linear system
            try:
                w_flat, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            W = w_flat.reshape(pn, R)
            
            if verify(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
                
        return None
    
    # Try ranks from 12 to 17 with adaptive time budget
    for R in range(12, 18):
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 5.0:
            break
        time_budget = min(15.0, remaining - 5.0)
        result = try_find_rank(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
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
