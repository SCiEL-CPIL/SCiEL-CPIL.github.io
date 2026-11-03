import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 9
    V_dim = m * p  # 12
    W_dim = p * n  # 12
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T, U_dim, V_dim, W_dim

def _check_exact(U, V, W, T, tol=1e-10):
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)

def _optimize_rank(T, rank, U_dim, V_dim, W_dim, max_iter=6000, lr=0.005, seed=42):
    rng = np.random.RandomState(seed)
    U = rng.randn(U_dim, rank) * 0.1
    V = rng.randn(V_dim, rank) * 0.1
    W = rng.randn(W_dim, rank) * 0.1
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_approx - T
        loss = np.sum(residual ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-14:
            break
        
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W) * 2
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W) * 2
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V) * 2
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return best_U, best_V, best_W, best_loss

def _try_find_decomposition(T, rank, U_dim, V_dim, W_dim, time_budget=6.0):
    start = time.time()
    best_loss = np.inf
    best = None
    
    for seed in range(300):
        if time.time() - start > time_budget:
            break
        U, V, W, loss = _optimize_rank(T, rank, U_dim, V_dim, W_dim, max_iter=3000, lr=0.005, seed=seed)
        if loss < best_loss:
            best_loss = loss
            best = (U, V, W, loss)
        if loss < 1e-12:
            return U, V, W, loss
    
    if best is not None and best[3] < 1e-10:
        return best
    return None

def construct():
    T, U_dim, V_dim, W_dim = _build_tensor()
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook fallback
    U_s = np.zeros((U_dim, schoolbook_rank))
    V_s = np.zeros((V_dim, schoolbook_rank))
    W_s = np.zeros((W_dim, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best = (U_s, V_s, W_s)
    best_rank = schoolbook_rank
    
    # Try ranks from 20 to 35, return first (lowest) valid
    for rank in range(20, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        result = _try_find_decomposition(T, rank, U_dim, V_dim, W_dim, time_budget=5.0)
        if result is not None:
            U, V, W, loss = result
            if loss < 1e-10:
                best = (U, V, W)
                best_rank = rank
                break  # Found the lowest rank in this range
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
