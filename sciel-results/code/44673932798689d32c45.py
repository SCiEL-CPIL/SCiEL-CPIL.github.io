import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
START_TIME = time.time()
DEADLINE = 145

def _remaining_time():
    return max(0, DEADLINE - (time.time() - START_TIME))

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _try_round(arr, tol=1e-6):
    arr_int = np.round(arr)
    arr_half = np.round(arr * 2) / 2.0
    
    err_int = np.sum((arr_int - arr)**2)
    err_half = np.sum((arr_half - arr)**2)
    
    if err_int < tol:
        return arr_int
    elif err_half < tol:
        return arr_half
    else:
        return None

def _search_decomposition(n, m, p, target_rank, max_restarts=100, time_budget=10):
    T_target = _build_tensor(n, m, p)
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    total_params = target_rank * (size_u + size_v + size_w)
    
    def objective(x):
        U = x[:target_rank * size_u].reshape(target_rank, size_u)
        V = x[target_rank * size_u : target_rank * (size_u + size_v)].reshape(target_rank, size_v)
        W = x[target_rank * (size_u + size_v):].reshape(target_rank, size_w)
        
        T_rec = np.einsum('ra, rb, rc -> abc', U, V, W)
        diff = T_rec - T_target
        return np.sum(diff * diff)

    def gradient(x):
        U = x[:target_rank * size_u].reshape(target_rank, size_u)
        V = x[target_rank * size_u : target_rank * (size_u + size_v)].reshape(target_rank, size_v)
        W = x[target_rank * (size_u + size_v):].reshape(target_rank, size_w)
        
        T_rec = np.einsum('ra, rb, rc -> abc', U, V, W)
        diff = T_rec - T_target
        
        grad_U = np.einsum('abc,rb,rc -> ra', diff, V, W)
        grad_V = np.einsum('abc,ra,rc -> rb', diff, U, W)
        grad_W = np.einsum('abc,ra,rb -> rc', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_error = float('inf')
    best_x = None
    
    start_time = time.time()
    for i in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        if _remaining_time() < 10:
            break
            
        x0 = np.random.uniform(-1, 1, total_params)
        
        try:
            # Use scipy's minimize with BFGS for better convergence
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                          options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_error:
                best_error = res.fun
                best_x = res.x
                if best_error < 1e-12:
                    break
        except Exception:
            continue
            
    if best_x is None or best_error > 1e-8:
        return None
        
    U = best_x[:target_rank * size_u].reshape(target_rank, size_u)
    V = best_x[target_rank * size_u : target_rank * (size_u + size_v)].reshape(target_rank, size_v)
    W = best_x[target_rank * (size_u + size_v):].reshape(target_rank, size_w)
    
    U_r = _try_round(U)
    V_r = _try_round(V)
    W_r = _try_round(W)
    
    if U_r is None or V_r is None or W_r is None:
        return None
        
    T_rec = np.einsum('ra, rb, rc -> abc', U_r, V_r, W_r)
    if np.allclose(T_rec, T_target, atol=1e-9):
        return U_r, V_r, W_r
        
    return None

def _build_strassen_3x3():
    # Strassen-like decomposition for 3x3x3 (rank 23)
    # This is a known decomposition, but I'll use a simpler approach:
    # Just return the trivial 27-term decomposition for safety
    n, m, p = 3, 3, 3
    U = np.zeros((n*m, n*m*p), dtype=int)
    V = np.zeros((m*p, n*m*p), dtype=int)
    W = np.zeros((p*n, n*m*p), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    best_rank = n * m * p
    best_decomp = None
    
    # Strategy: Try to find a decomposition with rank < 45
    # Known lower bound: rank(3,3,3) = 19, rank(3,3,5) >= rank(3,3,3) = 19
    # Trivial: 45
    
    # First, try to find a rank-41 decomposition (23 + 18 structure)
    # or any rank < 45
    
    # Try ranks from 30 to 44
    for rank in range(30, 45):
        if _remaining_time() < 15:
            break
            
        # Allocate time based on rank
        if rank <= 35:
            time_budget = min(20, _remaining_time() - 5)
            max_restarts = 50
        elif rank <= 40:
            time_budget = min(15, _remaining_time() - 5)
            max_restarts = 30
        else:
            time_budget = min(10, _remaining_time() - 5)
            max_restarts = 20
        
        res = _search_decomposition(n, m, p, rank, max_restarts=max_restarts, time_budget=time_budget)
        
        if res is not None:
            U, V, W = res
            best_rank = rank
            best_decomp = (U, V, W)
            break
            
    if best_decomp is None:
        # Fallback to schoolbook (rank 45)
        U = np.zeros((n*m, n*m*p), dtype=int)
        V = np.zeros((m*p, n*m*p), dtype=int)
        W = np.zeros((p*n, n*m*p), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                    idx += 1
        best_decomp = (U, V, W)
        best_rank = n * m * p
        
    U, V, W = best_decomp
    
    def to_list(arr):
        out = []
        for row in arr:
            row_list = []
            for val in row:
                if val == int(val):
                    row_list.append(int(val))
                else:
                    row_list.append(float(val))
            out.append(row_list)
        return out
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
