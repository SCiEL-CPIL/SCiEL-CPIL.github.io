import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify(U, V, W):
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            if len(row) != rank:
                return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _loss_and_grad(x, n, m, p, rank):
    """Vectorized loss and gradient using einsum."""
    U = x[0:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Build index arrays for the tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need to compute this for all i,j,k
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Using einsum: U3 is (n,m,r), V3 is (m,p,r), W3 is (p,n,r)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    residual = T - 1.0
    loss = np.sum(residual ** 2)
    
    # Gradients
    # dL/dU3[i,j,r] = 2 * sum_{k} residual[i,j,k] * V3[j,k,r] * W3[k,i,r]
    # dL/dV3[j,k,r] = 2 * sum_{i} residual[i,j,k] * U3[i,j,r] * W3[k,i,r]
    # dL/dW3[k,i,r] = 2 * sum_{j} residual[i,j,k] * U3[i,j,r] * V3[j,k,r]
    
    grad_U3 = 2.0 * np.einsum('ijk,jkr,kir->ijr', residual, V3, W3)
    grad_V3 = 2.0 * np.einsum('ijk,ijr,kir->jkr', residual, U3, W3)
    grad_W3 = 2.0 * np.einsum('ijk,ijr,jkr->kir', residual, U3, V3)
    
    grad_U = grad_U3.reshape(n*m, rank)
    grad_V = grad_V3.reshape(m*p, rank)
    grad_W = grad_W3.reshape(p*n, rank)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return loss, grad

def _try_rank(rank, time_limit=8.0, max_restarts=4):
    n, m, p = 3, 3, 6
    total_vars = (n*m + m*p + p*n) * rank
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(restart * 7919 + 42)
        
        # Initialize with schoolbook-like structure plus noise
        # Use first 'rank' terms of schoolbook as base
        U0 = np.zeros((n*m, rank))
        V0 = np.zeros((m*p, rank))
        W0 = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U0[i * m + j, col] = 1.0
                    V0[j * p + k, col] = 1.0
                    W0[k * n + i, col] = 1.0
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        
        # Add small random perturbation to escape symmetry
        noise_scale = 0.1
        U0 += noise_scale * rng.randn(n*m, rank)
        V0 += noise_scale * rng.randn(m*p, rank)
        W0 += noise_scale * rng.randn(p*n, rank)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(_loss_and_grad, x0, args=(n, m, p, rank), method='L-BFGS-B',
                           jac=True, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                                              'maxls': 50, 'maxfun': 5000})
            
            if res.fun < 1e-10:
                U_flat = res.x[0:n*m*rank].reshape(n*m, rank)
                V_flat = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                U_list = U_flat.tolist()
                V_list = V_flat.tolist()
                W_list = W_flat.tolist()
                
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            pass
            
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Leave 10s safety margin
    
    # Try ranks from 53 down, with decreasing time budgets
    # The schoolbook is 54, so we start at 53
    for rank in range(53, 44, -1):
        if time.time() > deadline - 5.0:
            break
        remaining = deadline - time.time() - 2.0
        if remaining < 3.0:
            break
        
        time_limit = min(8.0, remaining)
        max_restarts = max(2, int(remaining / 3.0))
        
        result = _try_rank(rank, time_limit=time_limit, max_restarts=max_restarts)
        if result is not None:
            return result
            
    return _schoolbook()
