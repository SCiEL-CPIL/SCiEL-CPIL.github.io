# EVOLVE-BLOCK-START
import random
import math
import numpy as np
from itertools import product

def _build_schoolbook():
    """Standard schoolbook matrix multiplication tensor decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(result, ntest=3, seed=42):
    """Validate that the decomposition computes exact matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    if not U or not V or not W:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Check exactness using integer arithmetic
    rng = random.Random(seed)
    for t in range(ntest):
        A = np.array([[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)])
        B = np.array([[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)])
        Ctrue = A @ B
        
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        Ccomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            x_r = np.sum(U_reshaped[:, :, r] * A[:, :, np.newaxis], axis=1) # n
            y_r = np.sum(V_reshaped[:, :, r] * B[:, :, np.newaxis], axis=0) # p
            Ccomp += (x_r[:, np.newaxis] * y_r[np.newaxis, :]) * W_reshaped[:, :, r].T
        
        if not np.allclose(Ccomp, Ctrue, atol=1e-9):
            return False
    return True

def _snapped_result(U, V, W):
    """Snap values to nearest 0.5 and return dict."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return {"U": U_s, "V": V_s, "W": W_s}

def _local_search_numpy(U, V, W, max_iters=2000, time_limit=None):
    """Local search using NumPy for speed."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    def _error():
        T_hat = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.sum((T_hat - 1.0)**2)
        
    current_err = _error()
    if current_err < 1e-9:
        return U, V, W, current_err
        
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
            
        mat_idx = np.random.randint(3)
        if mat_idx == 0:
            i, j, r = np.random.randint(n), np.random.randint(m), np.random.randint(rank)
            old_val = U_reshaped[i, j, r]
            new_val = old_val + np.random.choice([-1.0, 1.0])
            U_reshaped[i, j, r] = new_val
            new_err = _error()
            if new_err < current_err:
                current_err = new_err
            else:
                U_reshaped[i, j, r] = old_val
        elif mat_idx == 1:
            j, k, r = np.random.randint(m), np.random.randint(p), np.random.randint(rank)
            old_val = V_reshaped[j, k, r]
            new_val = old_val + np.random.choice([-1.0, 1.0])
            V_reshaped[j, k, r] = new_val
            new_err = _error()
            if new_err < current_err:
                current_err = new_err
            else:
                V_reshaped[j, k, r] = old_val
        else:
            k, i, r = np.random.randint(p), np.random.randint(n), np.random.randint(rank)
            old_val = W_reshaped[k, i, r]
            new_val = old_val + np.random.choice([-1.0, 1.0])
            W_reshaped[k, i, r] = new_val
            new_err = _error()
            if new_err < current_err:
                current_err = new_err
            else:
                W_reshaped[k, i, r] = old_val
                
        if current_err < 1e-9:
            break
            
    return U, V, W, current_err

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    budget = _TIME_LIMIT - elapsed
    if budget <= 10:
        best = _build_schoolbook()
        return {
            "U": best['U'].tolist(),
            "V": best['V'].tolist(),
            "W": best['W'].tolist()
        }
        
    best = _build_schoolbook()
    best_rank = best['U'].shape[1]
    
    ranks_to_try = [40, 48, 56, 64, 72, 80, 88]
    
    for target_rank in ranks_to_try:
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        if target_rank >= best_rank:
            continue
            
        # Try a few seeds
        for seed in range(3):
            if time.time() - _START_TIME > _TIME_LIMIT - 15:
                break
                
            np.random.seed(seed + 100)
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Sparse init
            for r in range(target_rank):
                for _ in range(n * m // 2):
                    idx = np.random.randint(n * m)
                    U[idx, r] = np.random.choice([-1.0, 1.0])
                for _ in range(m * p // 2):
                    idx = np.random.randint(m * p)
                    V[idx, r] = np.random.choice([-1.0, 1.0])
                for _ in range(p * n // 2):
                    idx = np.random.randint(p * n)
                    W[idx, r] = np.random.choice([-1.0, 1.0])
                    
            U, V, W, err = _local_search_numpy(U, V, W, max_iters=1000, time_limit=time.time() + 3)
            
            U_s, V_s, W_s = _snapped_result(U, V, W)['U'], _snapped_result(U, V, W)['V'], _snapped_result(U, V, W)['W']
            
            U_reshaped = U_s.reshape(n, m, -1)
            V_reshaped = V_s.reshape(m, p, -1)
            W_reshaped = W_s.reshape(p, n, -1)
            T_hat = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
            snapped_err = np.sum((T_hat - 1.0)**2)
            
            if snapped_err < 1e-9:
                if _validate_decomposition({"U": U_s, "V": V_s, "W": W_s}):
                    best = {"U": U_s, "V": V_s, "W": W_s}
                    best_rank = target_rank
                    break # Found valid, try lower? No, we iterate upwards. So break out of loop?
        
    return {
        "U": best['U'].tolist(),
        "V": best['V'].tolist(),
        "W": best['W'].tolist()
    }
# EVOLVE-BLOCK-END
