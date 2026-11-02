import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    from scipy.optimize import least_squares

    def residuals(params, rank):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
        W = params[(n*m+m*p)*rank:].reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], np.multiply(V[idx_v, :], W[idx_w, :]))
                    res.append(val - 1.0)
        return res

    # Search for lower ranks
    # We try 74, 73, 72
    for rank in [74, 73, 72]:
        if time.time() > deadline:
            break
        
        # Fewer restarts for lower ranks as they are harder
        restarts = 3 if rank == 74 else 2
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with small variance
            init = np.random.randn((n*m + m*p + p*n) * rank) * 0.1
            
            try:
                sol = least_squares(residuals, init, args=(rank,), method='trf', max_nfev=1000)
                
                if sol.cost < 1e-12:
                    U = sol.x[:n*m*rank].reshape(n*m, rank)
                    V = sol.x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
                    W = sol.x[(n*m+m*p)*rank:].reshape(p*n, rank)
                    
                    # Round to 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.dot(U_r[idx_u, :], np.multiply(V_r[idx_v, :], W_r[idx_w, :]))
                                if abs(val - 1.0) > 1e-9:
                                    exact = False
                                    break
                        if not exact: break
                    if exact:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = rank
                        break
            except Exception:
                pass

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
