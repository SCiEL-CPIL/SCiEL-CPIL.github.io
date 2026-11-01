import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)
    # W: (p*n, rank) = (6, rank)
    
    # Precompute indices
    i_idx = np.repeat(np.arange(n), m*p) # Not quite, let's use meshgrid
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    ii = ii.flatten()
    jj = jj.flatten()
    kk = kk.flatten()
    
    # Off-diagonal indices
    off_diag = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    for ip in range(n):
                        if (kp, ip) != (k, i):
                            off_diag.append((i, j, k, kp, ip))
    off_diag = np.array(off_diag)
    
    def compute_residuals(x, rank):
        U_flat = x[:n*m*rank].reshape(n*m, rank)
        V_flat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        # Diagonal residuals: sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        res_diag = np.zeros(len(ii))
        for idx in range(len(ii)):
            i, j, k = ii[idx], jj[idx], kk[idx]
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            s = np.dot(U_flat[u_idx], V_flat[v_idx] * W_flat[w_idx])
            res_diag[idx] = s - 1.0
            
        # Off-diagonal residuals: sum_r U[i,j,r] V[j,k,r] W[kp,ip,r]
        res_off = np.zeros(len(off_diag))
        for idx in range(len(off_diag)):
            i, j, k, kp, ip = off_diag[idx]
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = kp*n + ip
            s = np.dot(U_flat[u_idx], V_flat[v_idx] * W_flat[w_idx])
            res_off[idx] = s
            
        return np.concatenate([res_diag, res_off])
    
    def objective(x, rank):
        res = compute_residuals(x, rank)
        return np.sum(res**2)
    
    def is_half_integer(arr, tol=1e-9):
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
        
    def verify_solution(U, V, W, tol=1e-6):
        if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
            return False
        res = compute_residuals(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), U.shape[1])
        return np.sum(res**2) < tol

    # Schoolbook as fallback
    def schoolbook():
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
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def try_rank(target_rank, max_attempts=20):
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return False
            
            np.random.seed(42 + target_rank * 1000 + attempt)
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            try:
                res_opt = minimize(objective, x0, args=(target_rank,), method='BFGS', 
                                   options={'maxiter': 200, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res_opt.fun < 1e-6:
                U_sol = res_opt.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = res_opt.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = res_opt.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Try rounding
                U_r = round_to_half(U_sol)
                V_r = round_to_half(V_sol)
                W_r = round_to_half(W_sol)
                
                if verify_solution(U_r, V_r, W_r, 1e-6):
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = target_rank
                    return True
                elif verify_solution(U_sol, V_sol, W_sol, 1e-6):
                    # If continuous solution works but not half-integer, 
                    # it's not valid for the task (must be half-integer).
                    # But maybe we can find a nearby half-integer solution?
                    pass
                    
        return False

    # Try ranks from low to high
    # Known rank for 2x3x3 is 16.
    for target_rank in range(16, rank_sb):
        if time.time() > deadline - 15:
            break
        found = try_rank(target_rank, max_attempts=50)
        if found:
            # Found a solution, try to go lower if time permits?
            # No, we iterate from low to high, so first found is minimal.
            break
            
    return best
