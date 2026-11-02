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

def _verify(U, V, W, n=3, m=3, p=6):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
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
    U = x[0:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                residual = s - 1.0
                loss += residual * residual
                for r in range(rank):
                    grad_U[i*m + j, r] += 2 * residual * V[j*p + k, r] * W[k*n + i, r]
                    grad_V[j*p + k, r] += 2 * residual * U[i*m + j, r] * W[k*n + i, r]
                    grad_W[k*n + i, r] += 2 * residual * U[i*m + j, r] * V[j*p + k, r]
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return loss, grad

def _try_rank(n, m, p, rank, time_limit=10.0, max_restarts=5):
    total_vars = (n*m + m*p + p*n) * rank
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        rng = np.random.RandomState(restart + 12345)
        x0 = rng.choice([0, 0.5, -0.5, 1, -1], size=total_vars)
        
        try:
            res = minimize(_loss_and_grad, x0, args=(n, m, p, rank), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success and res.fun < 1e-10:
                U_flat = res.x[0:n*m*rank].reshape(n*m, rank)
                V_flat = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                U_rounded = np.round(U_flat * 2) / 2.0
                V_rounded = np.round(V_flat * 2) / 2.0
                W_rounded = np.round(W_flat * 2) / 2.0
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            pass
            
    return None

def _construct_3x3x6_from_3x3x3(decomp_333):
    """
    Construct a 3x3x6 decomposition from a 3x3x3 decomposition.
    decomp_333: dict with U, V, W for 3x3x3.
    """
    n, m, p_small = 3, 3, 3
    p_large = 6
    rank_small = len(decomp_333['U'][0])
    rank_large = 2 * rank_small
    
    U_small = decomp_333['U']
    V_small = decomp_333['V']
    W_small = decomp_333['W']
    
    U_large = [[0] * rank_large for _ in range(n * m)]
    V_large = [[0] * rank_large for _ in range(m * p_large)]
    W_large = [[0] * rank_large for _ in range(p_large * n)]
    
    # Block 1: k in 0..2
    for i in range(n):
        for j in range(m):
            for r in range(rank_small):
                U_large[i * m + j][r] = U_small[i * m + j][r]
                
    for j in range(m):
        for k in range(p_small):
            for r in range(rank_small):
                V_large[j * p_large + k][r] = V_small[j * p_small + k][r]
                
    for k in range(p_small):
        for i in range(n):
            for r in range(rank_small):
                W_large[k * n + i][r] = W_small[k * n + i][r]
                
    # Block 2: k in 3..5
    offset = rank_small
    for i in range(n):
        for j in range(m):
            for r in range(rank_small):
                U_large[i * m + j][offset + r] = U_small[i * m + j][r]
                
    for j in range(m):
        for k in range(p_small):
            for r in range(rank_small):
                V_large[j * p_large + (k + p_small)][offset + r] = V_small[j * p_small + k][r]
                
    for k in range(p_small):
        for i in range(n):
            for r in range(rank_small):
                W_large[(k + p_small) * n + i][offset + r] = W_small[k * n + i][r]
                
    return {"U": U_large, "V": V_large, "W": W_large}

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Try to find a 3x3x3 decomposition with rank < 27
    # If found, we can build a 3x3x6 decomposition with rank 2*rank_333
    # Schoolbook 3x3x3 is 27. So any rank < 27 is an improvement.
    
    # Try ranks 23, 24, 25, 26 for 3x3x3
    for rank_333 in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
            
        decomp_333 = _try_rank(3, 3, 3, rank_333, time_limit=20.0, max_restarts=5)
        if decomp_333 is not None:
            decomp_336 = _construct_3x3x6_from_3x3x3(decomp_333)
            if _verify(decomp_336['U'], decomp_336['V'], decomp_336['W'], 3, 3, 6):
                return decomp_336
                
    # Fallback to direct search for 3x3x6 with rank < 54
    # This is less likely to succeed but worth a try if time remains
    for rank in range(53, 46, -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank(3, 3, 6, rank, time_limit=5.0, max_restarts=2)
        if result is not None:
            return result
            
    return _schoolbook()
