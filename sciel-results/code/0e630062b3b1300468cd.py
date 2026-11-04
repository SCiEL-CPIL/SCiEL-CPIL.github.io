# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Precompute index arrays for fast verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def compute_residual(U, V, W, rank):
        """Compute residual vector."""
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        return total - 1.0
    
    def check_decomposition(U, V, W, rank, tol=1e-8):
        residual = compute_residual(U, V, W, rank)
        return np.max(np.abs(residual)) <= tol

    # Value set for snapping
    value_set = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])

    def snap_to_value_set(M):
        """Snap matrix entries to nearest value in value_set."""
        M_snapped = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_snapped[i, j] = value_set[np.argmin(np.abs(value_set - M[i, j]))]
        return M_snapped

    # Try to find a decomposition using smooth optimization
    # Start with rank 64 (based on 2*23 + 18 estimate) and go down
    for rank in range(64, 40, -2):
        if time.time() > deadline - 20:
            break
            
        # Number of attempts
        max_attempts = 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize U, V, W
            # Use structured init: random but dense
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Objective function: sum of squares of residual
            def objective(params):
                U_flat = params[:n * m * rank].reshape(n * m, rank)
                V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
                res = compute_residual(U_flat, V_flat, W_flat, rank)
                return np.sum(res * res)
            
            # Initial params
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            try:
                res_opt = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
                if res_opt.success or res_opt.fun < 1e-6:
                    U_opt = res_opt.x[:n * m * rank].reshape(n * m, rank)
                    V_opt = res_opt.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                    W_opt = res_opt.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                    
                    # Snap to value set
                    U_snap = snap_to_value_set(U_opt)
                    V_snap = snap_to_value_set(V_opt)
                    W_snap = snap_to_value_set(W_opt)
                    
                    if check_decomposition(U_snap, V_snap, W_snap, rank, tol=1e-6):
                        best_rank = rank
                        best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        return best_result
            except Exception:
                pass
                
    # Fallback: return schoolbook
    return best_result
# EVOLVE-BLOCK-END
