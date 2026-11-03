"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def compute_residual(U, V, W):
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        R -= T
        return R
    
    def objective_and_grad(params, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n * m, r)
        V = params[size_U:size_U + size_V].reshape(m * p, r)
        W = params[size_U + size_V:].reshape(p * n, r)
        
        R = compute_residual(U, V, W)
        loss = np.sum(R ** 2)
        
        grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V, W)
        grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U, W)
        grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def check_half_integer_exact(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if np.max(np.abs(U - U_r)) > 1e-6:
            return False, None, None, None
        if np.max(np.abs(V - V_r)) > 1e-6:
            return False, None, None, None
        if np.max(np.abs(W - W_r)) > 1e-6:
            return False, None, None, None
        
        R = compute_residual(U_r, V_r, W_r)
        if np.max(np.abs(R)) < 1e-9:
            return True, U_r, V_r, W_r
        return False, None, None, None
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Search ranks from low to high
    # Lower bound: for 3x3x3 it's 23, for 3x3x5 estimate ~30-35
    min_rank = 25
    max_rank = n * m * p  # 45
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Number of restarts based on remaining time
        remaining = deadline - time.time()
        num_restarts = max(3, int(remaining / 2))
        num_restarts = min(num_restarts, 20)
        
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(12345 + restart * 1000 + r)
            params = np.random.randn(r * (n * m + m * p + p * n)) * 0.1
            
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, r)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, r)[1],
                    method='L-BFGS-B',
                    options={'gtol': 1e-10, 'maxiter': 500, 'maxfun': 2000}
                )
                
                if res.fun < 1e-8:
                    size_U = n * m * r
                    size_V = m * p * r
                    U = res.x[:size_U].reshape(n * m, r)
                    V = res.x[size_U:size_U + size_V].reshape(m * p, r)
                    W = res.x[size_U + size_V:].reshape(p * n, r)
                    
                    is_exact, U_r, V_r, W_r = check_half_integer_exact(U, V, W)
                    if is_exact:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            except Exception:
                continue
        
        if found:
            break
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
