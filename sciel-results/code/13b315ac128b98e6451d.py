import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_fast(U, V, W, n, m, p, T_target):
    """Fast verification of tensor decomposition."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check half-integer constraint
    if not np.allclose(U_np * 2, np.round(U_np * 2), atol=1e-9):
        return False
    if not np.allclose(V_np * 2, np.round(V_np * 2), atol=1e-9):
        return False
    if not np.allclose(W_np * 2, np.round(W_np * 2), atol=1e-9):
        return False
    
    # Reconstruct and compare
    T_recon = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
    return True

def _try_remove_columns(U, V, W, n, m, p, T_target, deadline):
    """Try to remove redundant columns from the decomposition."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        rank = U_arr.shape[1]
        for c in range(rank):
            if time.time() > deadline - 5:
                break
            # Try removing column c
            U_test = np.delete(U_arr, c, axis=1)
            V_test = np.delete(V_arr, c, axis=1)
            W_test = np.delete(W_arr, c, axis=1)
            
            if U_test.shape[1] == 0:
                continue
                
            T_recon = np.einsum('uc,vc,wc->uvw', U_test, V_test, W_test)
            if np.allclose(T_recon, T_target, atol=1e-9):
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                changed = True
                break
    
    return U_arr, V_arr, W_arr

def _gradient_descent(U0, V0, W0, T_target, n, m, p, target_rank, deadline):
    """Gradient descent to find a lower-rank decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with random values
    U = np.random.randn(nm, target_rank) * 0.1
    V = np.random.randn(mp, target_rank) * 0.1
    W = np.random.randn(pn, target_rank) * 0.1
    
    lr = 0.01
    best_err = np.inf
    
    for it in range(500):
        if time.time() > deadline - 5:
            break
        
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        E = T_recon - T_target
        err = np.linalg.norm(E)
        
        if err < best_err:
            best_err = err
            U_best = U.copy()
            V_best = V.copy()
            W_best = W.copy()
        
        if err < 1e-6:
            break
        
        grad_U = np.einsum('uvw,vc,wc->uc', E, V, W)
        grad_V = np.einsum('uvw,uc,wc->vc', E, U, W)
        grad_W = np.einsum('uvw,uc,vc->wc', E, U, V)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Try rounding to half-integers periodically
        if it % 50 == 0 and err < 0.1:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_fast(U_r, V_r, W_r, n, m, p, T_target):
                return U_r, V_r, W_r
    
    # Check best found
    if best_err < 1e-6:
        U_r = np.round(U_best * 2) / 2
        V_r = np.round(V_best * 2) / 2
        W_r = np.round(W_best * 2) / 2
        if _verify_fast(U_r, V_r, W_r, n, m, p, T_target):
            return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    T_target = _build_target_tensor(n, m, p)
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank by removing columns
    U_reduced, V_reduced, W_reduced = _try_remove_columns(
        U.tolist(), V.tolist(), W.tolist(), n, m, p, T_target, deadline
    )
    
    if U_reduced.shape[1] < rank_schoolbook:
        return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    
    # Try gradient descent for lower ranks
    for r in range(30, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        result = _gradient_descent(None, None, None, T_target, n, m, p, r, deadline)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
