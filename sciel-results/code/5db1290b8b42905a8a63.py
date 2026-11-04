import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
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

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_als(r, time_budget):
        start = time.time()
        
        for seed in range(30):
            if time.time() - start > time_budget:
                break
            np.random.seed(seed * 31 + 7)
            
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for it in range(20):
                if time.time() - start > time_budget:
                    break
                
                # Solve W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = W_col
                        except:
                            return None
                            
                # Solve U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        try:
                            U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = U_col
                        except:
                            return None
                            
                # Solve V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = V_col
                        except:
                            return None
                            
                # Check loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            loss += (val - 1.0)**2
                
                if loss < 1e-10:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None

    # Try to find a lower rank; unlikely but fast to check
    for r in [71, 70, 69]:
        remaining = timeout - (time.time() - start_time)
        if remaining < 3:
            break
        budget = min(remaining - 2, 8.0)
        if budget > 0:
            res = try_als(r, budget)
            if res:
                return res

    return schoolbook()
