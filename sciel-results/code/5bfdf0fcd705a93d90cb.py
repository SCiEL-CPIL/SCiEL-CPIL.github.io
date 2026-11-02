import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 36)
    rank_sb = n * m * p
    U = np.zeros((n*m, rank_sb), dtype=float)
    V = np.zeros((m*p, rank_sb), dtype=float)
    W = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_sb
    
    # Try to find lower-rank decomposition
    for R in range(18, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(300):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 7919 + 13)
            U_r = np.random.randn(n*m, R)
            V_r = np.random.randn(m*p, R)
            W_r = np.random.randn(p*n, R)
            
            for it in range(40):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W (fix U, V)
                W_new = np.zeros((p*n, R))
                ok = True
                for k in range(p):
                    for i in range(n):
                        A = np.array([[U_r[i*m+j, r] * V_r[j*p+k, r] for r in range(R)] for j in range(m)])
                        b = np.ones(m)
                        W_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ W_row, b, atol=1e-12):
                            ok = False
                            break
                        W_new[k*n+i] = W_row
                    if not ok:
                        break
                if not ok:
                    break
                W_r = W_new
                
                # Solve for U (fix V, W)
                U_new = np.zeros((n*m, R))
                ok = True
                for i in range(n):
                    for j in range(m):
                        A = np.array([[V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(R)] for k in range(p)])
                        b = np.ones(p)
                        U_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ U_row, b, atol=1e-12):
                            ok = False
                            break
                        U_new[i*m+j] = U_row
                    if not ok:
                        break
                if not ok:
                    break
                U_r = U_new
                
                # Solve for V (fix U, W)
                V_new = np.zeros((m*p, R))
                ok = True
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U_r[i*m+j, r] * W_r[k*n+i, r] for r in range(R)] for i in range(n)])
                        b = np.ones(n)
                        V_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ V_row, b, atol=1e-12):
                            ok = False
                            break
                        V_new[j*p+k] = V_row
                    if not ok:
                        break
                if not ok:
                    break
                V_r = V_new
                
                # Check all constraints
                max_err = max(
                    abs(float(np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])) - 1.0)
                    for i in range(n) for j in range(m) for k in range(p)
                )
                if max_err < 1e-10:
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    return best
