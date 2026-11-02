import numpy as np
from scipy.optimize import minimize

def _find_decomposition(n, m, p, R, max_iter=100, n_restarts=5):
    def error(params):
        U = params[:R*n*m].reshape(R, n*m)
        V = params[R*n*m:R*(n*m+m*p)].reshape(R, m*p)
        W = params[R*(n*m+m*p):].reshape(R, p*n)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i*m + j
                    vj = j*p + k
                    wk = k*n + i
                    val = np.sum(U[:, ui] * V[:, vj] * W[:, wk])
                    err += (val - 1.0)**2
        return err

    best_params = None
    best_err = np.inf
    
    for _ in range(n_restarts):
        params = np.random.randn(R * (n*m + m*p + p*n))
        res = minimize(error, params, method='L-BFGS-B', options={'maxiter': max_iter})
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
            if best_err < 1e-6:
                break
                
    if best_err < 1e-6:
        R_total = R * (n*m + m*p + p*n)
        best_params = np.round(best_params * 2) / 2
        
        U = best_params[:R*n*m].reshape(R, n*m).T
        V = best_params[R*n*m:R*(n*m+m*p)].reshape(R, m*p).T
        W = best_params[R*(n*m+m*p):].reshape(R, p*n).T
        
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i*m + j
                    vj = j*p + k
                    wk = k*n + i
                    val = np.sum(U[ui, :] * V[vj, :] * W[wk, :])
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        
        if ok:
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition for 2x3 by 3x3
    # Use it 3 times for 2x3 by 3x9
    
    best_U_sub = None
    best_V_sub = None
    best_W_sub = None
    best_rank_sub = 18 # Trivial
    
    for R in [17, 16, 15]:
        res = _find_decomposition(2, 3, 3, R, max_iter=200, n_restarts=10)
        if res is not None:
            best_U_sub, best_V_sub, best_W_sub = res
            best_rank_sub = R
            break
            
    if best_U_sub is not None:
        rank_sub = best_rank_sub
        rank_total = 3 * rank_sub
        
        U_full = np.zeros((n*m, rank_total))
        V_full = np.zeros((m*p, rank_total))
        W_full = np.zeros((p*n, rank_total))
        
        for block in range(3):
            offset = block * rank_sub
            
            U_full[:, offset:offset+rank_sub] = best_U_sub
            
            for j in range(m):
                for k_sub in range(3):
                    k_full = block * 3 + k_sub
                    V_full[j*p + k_full, offset:offset+rank_sub] = best_V_sub[j*3 + k_sub]
            
            for k_sub in range(3):
                k_full = block * 3 + k_sub
                for i in range(n):
                    W_full[k_full*n + i, offset:offset+rank_sub] = best_W_sub[k_sub*n + i]
            
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to trivial
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
