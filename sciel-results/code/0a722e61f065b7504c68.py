import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find a decomposition with rank < 24
    for r in range(20, 24):
        if time.time() > deadline:
            break
            
        res = try_find_decomposition(n, m, p, r, time_limit=30)
        if res is not None:
            # Verify
            U, V, W = res["U"], res["V"], res["W"]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return res
                
    return {"U": U_school, "V": V_school, "W": W_school}

def try_find_decomposition(n, m, p, rank, time_limit=10):
    start_time = time.time()
    num_restarts = 1000
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random U, V with half-integer entries
        U = np.random.randint(-4, 5, (n*m, rank)) * 0.5
        V = np.random.randint(-4, 5, (m*p, rank)) * 0.5
        
        # Build system for W
        A = np.zeros((n*m*p, p*n*rank))
        T = np.zeros(n*m*p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[idx] = 1.0
                    for c in range(rank):
                        A[idx, (k*n+i)*rank + c] = U[i*m+j, c] * V[j*p+k, c]
                    idx += 1
                    
        try:
            w, residuals, rank_sol, sv = np.linalg.lstsq(A, T, rcond=None)
        except:
            continue
            
        if residuals is not None and residuals[0] > 1e-9:
            continue
        if residuals is None:
            if np.linalg.norm(A @ w - T) > 1e-9:
                continue
                
        W = w.reshape(p*n, rank)
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        error += 1
                        break
                if error > 0: break
            if error > 0: break
        if error > 0:
            continue
            
        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
        
    return None
