import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 105)
    rank_school = n * m * p
    
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    deadline = time.time() + 115
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def err(U, V, W):
        e = 0.0
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                    e += (s - 1.0) ** 2
        return e
    
    # Try ranks from 104 down to 60
    for r in range(104, 59, -1):
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init in range(50):
            if time.time() > deadline:
                break
            
            # Strategy: start from schoolbook, zero out some columns, add noise
            np.random.seed(999 + init)
            
            if init % 2 == 0:
                # Start from schoolbook with noise
                U = U_school[:, :r].copy() + np.random.randn(n * m, r) * 0.2
                V = V_school[:, :r].copy() + np.random.randn(m * p, r) * 0.2
                W = W_school[:, :r].copy() + np.random.randn(p * n, r) * 0.2
            else:
                # Pure random
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
            
            curr = err(U, V, W)
            
            # Local search with adaptive step
            step = 0.5
            for _ in range(40):
                if time.time() > deadline:
                    break
                if curr < 1e-14:
                    break
                
                improved = False
                for _ in range(8):
                    idx = np.random.randint(0, 3)
                    if idx == 0:
                        U_new = U + np.random.randn(n * m, r) * step
                        new = err(U_new, V, W)
                        if new < curr:
                            U, curr = U_new, new
                            improved = True
                    elif idx == 1:
                        V_new = V + np.random.randn(m * p, r) * step
                        new = err(U, V_new, W)
                        if new < curr:
                            V, curr = V_new, new
                            improved = True
                    else:
                        W_new = W + np.random.randn(p * n, r) * step
                        new = err(U, V, W_new)
                        if new < curr:
                            W, curr = W_new, new
                            improved = True
                
                if improved:
                    step = max(step * 0.9, 0.05)
                else:
                    step *= 0.5
                    if step < 0.01:
                        break
            
            if curr < 1e-12 and verify(U, V, W):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
