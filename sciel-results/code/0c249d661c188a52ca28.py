import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()

def _check_time(deadline):
    return time.time() > deadline

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            U_row = U_arr[:, r].reshape(n, m)
            term_U = (U_row * A).sum(axis=1)
            
            V_col = V_arr[:, r].reshape(m, p)
            term_V = (V_col * B).sum(axis=0)
            
            W_mat = W_arr[:, r].reshape(p, n)
            C_computed += W_mat.T * np.outer(term_U, term_V)
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _objective_and_grad(params, n, m, p, rank):
    U = params[0: n*m*rank].reshape(n, m, rank)
    V = params[n*m*rank: n*m*rank + m*p*rank].reshape(m, p, rank)
    W = params[n*m*rank + m*p*rank:].reshape(p, n, rank)
    
    # Compute S_{i, j, j', k}
    S = np.einsum('ijr,jkr,kir->ijjk', U, V, W)
    
    # Target T_{i, j, j', k} = delta_{jj'}
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i, j, j, k] = 1.0
                
    diff = S - T
    obj = np.sum(diff * diff)
    
    # Gradients
    grad_U = 2.0 * np.einsum('ijjk,jkr,kir->ijr', diff, V, W)
    grad_V = 2.0 * np.einsum('ijjk,ijr,kir->jkr', diff, U, W)
    grad_W = 2.0 * np.einsum('ijjk,ijr,jkr->kir', diff, U, V)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return obj, grad

def _find_decomposition(n, m, p, rank, seed=0, max_iters=2000, tol=1e-12):
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n, m, rank) * 0.1
    V = np.random.randn(m, p, rank) * 0.1
    W = np.random.randn(p, n, rank) * 0.1
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    res = minimize(
        _objective_and_grad, 
        params, 
        args=(n, m, p, rank),
        method='L-BFGS-B', 
        jac=True,
        options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol, 'disp': False}
    )
    
    if res.fun < 1e-10:
        U_opt = res.x[0: n*m*rank].reshape(n, m, rank)
        V_opt = res.x[n*m*rank: n*m*rank + m*p*rank].reshape(m, p, rank)
        W_opt = res.x[n*m*rank + m*p*rank:].reshape(p, n, rank)
        
        U_list = U_opt.reshape(n*m, rank).tolist()
        V_list = V_opt.reshape(m*p, rank).tolist()
        W_list = W_opt.reshape(p*n, rank).tolist()
        
        return U_list, V_list, W_list, True
        
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 115
    
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 40 to 59
    for target_rank in range(40, 60):
        if _check_time(deadline - 10):
            break
            
        found = False
        for seed in range(20):
            if _check_time(deadline - 5):
                break
                
            U_list, V_list, W_list, converged = _find_decomposition(
                n, m, p, target_rank, seed=seed, max_iters=1000, tol=1e-10
            )
            
            if converged:
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
            
    return best
