import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

_DEADLINE_FRACTION = 0.85
_SEARCH_TIMEOUT = 100.0

def _build_target_tensor(n, m, p):
    T = np.ones((n, m, p), dtype=float)
    return T

def _residual(U, V, W, T_vec, n, m, p):
    rank = U.shape[1]
    T_full = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r]
        Vr = V[:, r]
        Wr = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i, j, k] += Ur[i * m + j] * Vr[j * p + k] * Wr[k * n + i]
    return np.linalg.norm(T_full.flatten() - T_vec)

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_int(arr):
    return np.round(arr * 2.0) / 2.0

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W

def _try_rank_continuous(n, m, p, rank, T_vec, deadline):
    if time.time() > deadline:
        return None
    
    best_residual = float('inf')
    best_factors = None
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_flat = params[:n * m * rank]
            V_flat = params[n * m * rank : n * m * rank + m * p * rank]
            W_flat = params[n * m * rank + m * p * rank :]
            
            U = U_flat.reshape(n * m, rank)
            V = V_flat.reshape(m * p, rank)
            W = W_flat.reshape(p * n, rank)
            
            return _residual(U, V, W, T_vec, n, m, p)
        
        # Use L-BFGS-B
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.fun < best_residual:
            best_residual = res.fun
            U_flat = res.x[:n * m * rank]
            V_flat = res.x[n * m * rank : n * m * rank + m * p * rank]
            W_flat = res.x[n * m * rank + m * p * rank :]
            U = U_flat.reshape(n * m, rank)
            V = V_flat.reshape(m * p, rank)
            W = W_flat.reshape(p * n, rank)
            best_factors = (U, V, W)
            
            # Snap to half-integers and check
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            if _verify_exact(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
                
    if best_factors is not None and best_residual < 1e-6:
        U, V, W = best_factors
        U_r = _round_to_half_int(U)
        V_r = _round_to_half_int(V)
        W_r = _round_to_half_int(W)
        if _verify_exact(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
            
    return None

def construct():
    t0 = time.time()
    hard_deadline = t0 + _SEARCH_TIMEOUT
    n, m, p = DIMENSIONS
    T_vec = _build_target_tensor(n, m, p).flatten()
    
    # Try ranks from 32 down to 27
    for rank in range(32, 26, -1):
        if time.time() > hard_deadline:
            break
        result = _try_rank_continuous(n, m, p, rank, T_vec, hard_deadline)
        if result is not None:
            return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
            
    # Fallback
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
