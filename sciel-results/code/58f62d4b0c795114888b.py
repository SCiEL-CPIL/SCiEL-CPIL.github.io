import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Build the rank 32 base solution
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
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            alpha = i*m + k
            for j in range(p):
                beta = k*p + j
                gamma = j*n + i
                T_target[alpha, beta, gamma] = 1.0
    
    def compute_residual(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return T_recon - T_target
    
    def objective(params):
        size_u = n*m * rank
        size_v = m*p * rank
        U = params[:size_u].reshape(n*m, rank)
        V = params[size_u:size_u+size_v].reshape(m*p, rank)
        W = params[size_u+size_v:].reshape(p*n, rank)
        res = compute_residual(U, V, W)
        return np.sum(res**2)
    
    # Verify base solution
    base_obj = objective(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
    
    # Try to find rank-31 solution by zeroing one column and optimizing
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_obj_val = base_obj
    
    for r_remove in range(rank):
        if time.time() > deadline - 20:
            break
        
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, r_remove] = 0
        V_test[:, r_remove] = 0
        W_test[:, r_remove] = 0
        
        params = np.concatenate([U_test.flatten(), V_test.flatten(), W_test.flatten()])
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                          options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-12})
            if res.fun < best_obj_val - 1e-8:
                best_obj_val = res.fun
                size_u = n*m * rank
                size_v = m*p * rank
                best_U = res.x[:size_u].reshape(n*m, rank)
                best_V = res.x[size_u:size_u+size_v].reshape(m*p, rank)
                best_W = res.x[size_u+size_v:].reshape(p*n, rank)
        except Exception:
            pass
    
    # Round to half-integers if possible
    def can_round(arr):
        return np.all(np.abs(2*arr - np.round(2*arr)) < 1e-6)
    
    if can_round(best_U) and can_round(best_V) and can_round(best_W):
        best_U = np.round(2*best_U) / 2
        best_V = np.round(2*best_V) / 2
        best_W = np.round(2*best_W) / 2
    
    # Check if we actually reduced rank
    # Count non-zero columns
    def count_rank(U, V, W):
        # A column contributes if any of U[:,r], V[:,r], W[:,r] is non-zero
        rank_count = 0
        for r in range(U.shape[1]):
            if (np.any(np.abs(U[:, r]) > 1e-6) or 
                np.any(np.abs(V[:, r]) > 1e-6) or 
                np.any(np.abs(W[:, r]) > 1e-6)):
                rank_count += 1
        return rank_count
    
    final_rank = count_rank(best_U, best_V, best_W)
    
    # If we found a better solution, trim to actual rank
    if final_rank < rank:
        # Find which columns are non-zero
        active_cols = []
        for r in range(rank):
            if (np.any(np.abs(best_U[:, r]) > 1e-6) or 
                np.any(np.abs(best_V[:, r]) > 1e-6) or 
                np.any(np.abs(best_W[:, r]) > 1e-6)):
                active_cols.append(r)
        
        best_U = best_U[:, active_cols]
        best_V = best_V[:, active_cols]
        best_W = best_W[:, active_cols]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
