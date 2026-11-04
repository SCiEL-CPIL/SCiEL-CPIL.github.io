"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T is of size (n*m) x (m*p) x (p*n) = 6 x 9 x 6.
    We seek rank-r decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    Returns dict with U (6 x r), V (9 x r), W (6 x r) where r is the rank.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = N, M, P
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    
    # Build the full tensor T[i,j,k]
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(u_dim):
        i_a = i // m
        j_a = i % m
        for j in range(v_dim):
            j_a2 = j // p
            k_b = j % p
            if j_a != j_a2:
                continue
            for k in range(w_dim):
                k_b2 = k // n
                i_c = k % n
                if i_a != i_c or k_b != k_b2:
                    continue
                T[i, j, k] = 1.0

    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_s = np.zeros((u_dim, rank_school))
    V_s = np.zeros((v_dim, rank_school))
    W_s = np.zeros((w_dim, rank_school))
    col = 0
    for i_a in range(n):
        for j_a in range(m):
            for k_b in range(p):
                i_idx = i_a * m + j_a
                j_idx = j_a * p + k_b
                k_idx = k_b * n + i_a
                U_s[i_idx, col] = 1.0
                V_s[j_idx, col] = 1.0
                W_s[k_idx, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school
    
    # Try to find rank 17 decomposition
    target_rank = 17
    
    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        T_flat = T.flatten()
        
        # Objective function
        def objective(params):
            U_r = params[:u_dim * rank].reshape(u_dim, rank)
            V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
            W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
            recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
            diff = recon.flatten() - T_flat
            return np.dot(diff, diff)
        
        # Try multiple initializations
        for trial in range(50):
            if time.time() > deadline:
                return None
            
            # Structured initialization: 
            # Try to separate the last column of B (indices 6,7,8 in V)
            # and handle it with 6 mults, and the rest with 11.
            # This is a heuristic.
            
            np.random.seed(42 + trial)
            U_init = np.random.randn(u_dim, rank) * 0.1
            V_init = np.random.randn(v_dim, rank) * 0.1
            W_init = np.random.randn(w_dim, rank) * 0.1
            
            # Add schoolbook structure to some columns
            if trial < 10:
                # Copy first 11 schoolbook columns
                U_init[:, :11] = U_s[:, :11]
                V_init[:, :11] = V_s[:, :11]
                W_init[:, :11] = W_s[:, :11]
                # Randomize the rest
                U_init[:, 11:] = np.random.randn(u_dim, rank-11) * 0.1
                V_init[:, 11:] = np.random.randn(v_dim, rank-11) * 0.1
                W_init[:, 11:] = np.random.randn(w_dim, rank-11) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                  options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
            
            if result.fun < 1e-6:
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                if np.max(np.abs(recon - T)) < 1e-5:
                    return U_r, V_r, W_r
        
        return None

    result = try_rank(target_rank)
    if result:
        best_U, best_V, best_W = result
        best_rank = target_rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
