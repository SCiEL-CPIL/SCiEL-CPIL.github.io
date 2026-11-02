import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    nm, mp, pn = n*m, m*p, p*n  # 6, 18, 12
    start_time = time.time()
    deadline = start_time + 100  # 100 second limit
    
    best_rank = n * m * p
    best_solution = None
    
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        B = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(rank)] for k in range(p)])
                        try:
                            U_new[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        B = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(rank)] for i in range(n)])
                        try:
                            V_new[j*p+k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        B = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(rank)] for j in range(m)])
                        try:
                            W_new[k*n+i, :] = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                
                U, V, W = U_new, V_new, W_new
                
                # Check validity
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-8:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    max_err_r = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                                max_err_r = max(max_err_r, abs(val - 1))
                    
                    if max_err_r < 1e-8:
                        best_rank = rank
                        best_solution = (U_r, V_r, W_r)
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
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
