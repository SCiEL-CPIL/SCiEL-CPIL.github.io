"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def fast_verify(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using einsum."""
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # result[i, jp, k, kp, ip, j] = sum_c U[i,j,c]*V[jp,k,c]*W[kp,ip,c]
    result = np.einsum('abc,dec,fgc->adefgb', U_r, V_r, W_r)
    
    # Target: 1 if j==jp and k==kp, else 0
    target = np.zeros((n, m, p, p, n, m))
    for k in range(p):
        for j in range(m):
            target[:, j, k, k, :, j] = 1.0
    
    return np.max(np.abs(result - target)) <= tol

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Try to find better decompositions with gradient-based optimization
    for init_rank in range(55, 44, -1):
        if time.time() - start_time > 125:
            break
        
        for restart in range(3):
            if time.time() - start_time > 125:
                break
            
            np.random.seed(restart * 1000 + init_rank)
            U = np.random.randn(nm, init_rank) * 0.1
            V = np.random.randn(mp, init_rank) * 0.1
            W = np.random.randn(pn, init_rank) * 0.1
            
            def objective_and_grad(params):
                U_f = params[:nm*init_rank].reshape(nm, init_rank)
                V_f = params[nm*init_rank:nm*init_rank+mp*init_rank].reshape(mp, init_rank)
                W_f = params[nm*init_rank+mp*init_rank:].reshape(pn, init_rank)
                
                U_r = U_f.reshape(n, m, init_rank)
                V_r = V_f.reshape(m, p, init_rank)
                W_r = W_f.reshape(p, n, init_rank)
                
                result = np.einsum('abc,dec,fgc->adefgb', U_r, V_r, W_r)
                
                target = np.zeros((n, m, p, p, n, m))
                for k in range(p):
                    for j in range(m):
                        target[:, j, k, k, :, j] = 1.0
                
                diff = result - target
                obj = np.sum(diff**2)
                
                # Gradients using einsum
                grad_U = 2 * np.einsum('adefgb,dec,fgc->abc', diff, V_r, W_r).reshape(nm, init_rank)
                grad_V = 2 * np.einsum('adefgb,abc,fgc->dec', diff, U_r, W_r).reshape(mp, init_rank)
                grad_W = 2 * np.einsum('adefgb,abc,dec->fgc', diff, U_r, V_r).reshape(pn, init_rank)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return obj, grad
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective_and_grad, x0, method='L-BFGS-B',
                             jac=True, options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if res.fun < 1e-6:
                    U_opt = res.x[:nm*init_rank].reshape(nm, init_rank)
                    V_opt = res.x[nm*init_rank:nm*init_rank+mp*init_rank].reshape(mp, init_rank)
                    W_opt = res.x[nm*init_rank+mp*init_rank:].reshape(pn, init_rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    if fast_verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                        best_rank = init_rank
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        break
                    elif fast_verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        best_rank = init_rank
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        break
            except Exception:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
