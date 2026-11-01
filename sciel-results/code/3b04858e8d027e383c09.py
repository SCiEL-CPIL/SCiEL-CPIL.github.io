import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _soft_max_abs(residual, beta=100.0):
    """Compute a smooth approximation of max(|residual|) using log-sum-exp."""
    abs_res = np.abs(residual)
    # Soft max via log-sum-exp: (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, subtract max
    max_val = np.max(abs_res)
    if max_val < 1e-15:
        return 0.0
    shifted = abs_res - max_val
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) + beta * max_val
    return log_sum_exp / beta

def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition by optimizing from rank-32 starting point."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank = 32
    
    # Precompute true tensor
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_score = _compute_residual(U0, V0, W0, n, m, p)
    
    # Try each column as the one to eliminate
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.85:
            break
        
        # Objective: minimize soft-max residual + lambda * (sum of squares of column target_col)
        def objective(params, lam=1.0, beta=100.0):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            # Compute decomposed tensor
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = M - T_true
            
            # Soft max of absolute residual
            smax = _soft_max_abs(residual, beta)
            
            # Penalty for column target_col
            col_penalty = np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2)
            
            return smax + lam * col_penalty
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Try multiple lambda values and beta values
        for lam in [0.01, 0.1, 0.5, 1.0, 5.0]:
            if time.time() - start_time > time_budget * 0.9:
                break
            
            for beta in [50.0, 100.0, 200.0]:
                if time.time() - start_time > time_budget * 0.92:
                    break
                
                try:
                    result = minimize(objective, x0, args=(lam, beta), method='L-BFGS-B', 
                                      options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8})
                except Exception:
                    continue
                
                U_flat = result.x[:nm * rank]
                V_flat = result.x[nm * rank : nm * rank + mp * rank]
                W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                U = U_flat.reshape(nm, rank)
                V = V_flat.reshape(mp, rank)
                W = W_flat.reshape(pn, rank)
                
                # Check if column is close to zero
                col_norm = np.sqrt(np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2))
                
                if col_norm < 0.05:
                    # Try rank-31 by dropping this column
                    U31 = np.delete(U, target_col, axis=1)
                    V31 = np.delete(V, target_col, axis=1)
                    W31 = np.delete(W, target_col, axis=1)
                    score31 = _compute_residual(U31, V31, W31, n, m, p)
                    if score31 < 1e-6:
                        # Final polish: try to reduce residual further
                        U31, V31, W31 = _polish_rank31(U31, V31, W31, n, m, p, T_true, 5.0)
                        score31_polished = _compute_residual(U31, V31, W31, n, m, p)
                        if score31_polished < 1e-6:
                            return U31, V31, W31, score31_polished
                
                # Update best
                cur_score = _compute_residual(U, V, W, n, m, p)
                if cur_score < best_score:
                    best_score = cur_score
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return None

def _polish_rank31(U, V, W, n, m, p, T_true, time_budget):
    """High-precision polish pass for rank-31 decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    start_time = time.time()
    
    def objective(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        
        # Use sum of squares for smooth optimization
        return np.sum(residual**2)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
    except Exception:
        return U, V, W
    
    if result.fun < 1e-10:
        U_flat = result.x[:nm * rank]
        V_flat = result.x[nm * rank : nm * rank + mp * rank]
        W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U_polished = U_flat.reshape(nm, rank)
        V_polished = V_flat.reshape(mp, rank)
        W_polished = W_flat.reshape(pn, rank)
        return U_polished, V_polished, W_polished
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        search_budget = min(remaining_time - 25, 100)  # Leave 25s safety margin
        result = _try_rank31_search(U, V, W, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-6:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Fall back to rank-32
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
