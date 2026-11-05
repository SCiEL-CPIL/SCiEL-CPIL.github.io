import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    Y = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    if j == j_prime:
                        Y[i, j, j_prime, k] = 1.0
    Y_flat = Y.flatten()

    def compute_error(params, rank):
        nm = n * m
        mp = m * p
        pn = p * n
        U = params[:nm * rank].reshape(n, m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m, p, rank)
        W = params[nm * rank + mp * rank:].reshape(p, n, rank)
        
        D = np.einsum('ijc,jkc,kic->ijjk', U, V, W)
        err = D.flatten() - Y_flat
        return np.sum(err**2)

    def get_matrices(params, rank):
        nm = n * m
        mp = m * p
        pn = p * n
        U = params[:nm * rank].reshape(n * m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m * p, rank)
        W = params[nm * rank + mp * rank:].reshape(p * n, rank)
        return U, V, W

    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    # Search for better rank
    # We limit the search time to ensure we return within deadline
    search_deadline = start_time + 120
    
    for rank in range(best_rank - 1, 19, -1):
        if time.time() > search_deadline:
            break
        
        found = False
        # Fewer attempts and iterations to speed up
        for attempt in range(2):
            if time.time() > search_deadline:
                break
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(compute_error, x0, args=(rank,), method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
                if res.fun < 1e-6:
                    U_opt, V_opt, W_opt = get_matrices(res.x, rank)
                    
                    # Check for half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify rounded
                    params_rounded = np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()])
                    err_rounded = compute_error(params_rounded, rank)
                    
                    if err_rounded < 1e-6:
                        best_rank = rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
                    else:
                        if res.fun < 1e-8:
                             best_rank = rank
                             best_U = U_opt
                             best_V = V_opt
                             best_W = W_opt
                             found = True
                             break
            except Exception:
                continue
        
        if found:
            continue
        else:
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
