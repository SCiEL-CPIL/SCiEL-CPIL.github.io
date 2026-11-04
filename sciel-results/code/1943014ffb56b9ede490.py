import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    diff = np.linalg.norm(T - T_recon)
    return diff < tol

def _round_to_half_int(M):
    return np.round(M * 2) / 2

def _is_half_integer(M, tol=1e-9):
    M2 = M * 2
    return np.all(np.abs(M2 - np.round(M2)) < tol)

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s kill
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, n * m * p))
    V_school = np.zeros((m * p, n * m * p))
    W_school = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Lower bound for rank
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upwards
    for r in range(lb, best_rank + 1):
        if time.time() > deadline - 25:
            break
        
        found = False
        
        # Multiple random restarts for this rank
        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(restart * 12345 + 67890)
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            prev_residual = np.inf
            
            # ALS iterations
            for it in range(80):
                if time.time() > deadline - 25:
                    break
                
                # Solve for U
                T_kr = T.reshape(n * m, m * p * p * n)
                M = np.zeros((r, m * p * p * n))
                for t in range(r):
                    M[t, :] = (V[:, t].reshape(-1, 1) * W[:, t].reshape(1, -1)).flatten()
                
                MMt = M @ M.T
                try:
                    U_new = T_kr @ M.T @ np.linalg.inv(MMt)
                except np.linalg.LinAlgError:
                    U_new = T_kr @ M.T @ np.linalg.pinv(MMt)
                
                # Solve for V
                T_kc = T.reshape(m * p, n * m * p * n)
                M_v = np.zeros((r, n * m * p * n))
                for t in range(r):
                    M_v[t, :] = (U_new[:, t].reshape(-1, 1) * W[:, t].reshape(1, -1)).flatten()
                
                MvMvt = M_v @ M_v.T
                try:
                    V_new = T_kc @ M_v.T @ np.linalg.inv(MvMvt)
                except np.linalg.LinAlgError:
                    V_new = T_kc @ M_v.T @ np.linalg.pinv(MvMvt)
                
                # Solve for W
                T_kb = T.reshape(p * n, n * m * m * p)
                M_w = np.zeros((r, n * m * m * p))
                for t in range(r):
                    M_w[t, :] = (U_new[:, t].reshape(-1, 1) * V_new[:, t].reshape(1, -1)).flatten()
                
                MwMwt = M_w @ M_w.T
                try:
                    W_new = T_kb @ M_w.T @ np.linalg.inv(MwMwt)
                except np.linalg.LinAlgError:
                    W_new = T_kb @ M_w.T @ np.linalg.pinv(MwMwt)
                
                # Calculate residual
                residual = np.linalg.norm(T - np.einsum('at,bt,ct->abc', U_new, V_new, W_new))
                
                if abs(prev_residual - residual) < 1e-12:
                    break
                
                prev_residual = residual
                U, V, W = U_new, V_new, W_new
            
            # Round to half-integers and verify
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                # Found a valid decomposition with rank r
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
    # If no better rank found, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
