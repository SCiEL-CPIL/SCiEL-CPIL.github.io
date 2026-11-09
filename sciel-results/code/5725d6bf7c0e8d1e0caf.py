import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Check if decomposition is valid using numpy. Returns True if valid."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j]
                    v_val = V[j*p + k]
                    w_val = W[k*n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_round_and_verify(U, V, W, rank):
        """Try to round to small rationals and verify."""
        for denom in [1, 2, 4, 8, 16]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            if verify(U_r, V_r, W_r, rank):
                return True, U_r, V_r, W_r
        return False, None, None, None

    # Initialize with schoolbook
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find lower rank decompositions
    # We will try ranks from schoolbook_rank - 1 down to a lower bound
    # Let's try up to 10 ranks below schoolbook
    for target_rank in range(schoolbook_rank - 1, max(30, schoolbook_rank - 10)):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(attempt * 100 + target_rank)
            # Initialize with small random values
            U_init = rng.randn(n*m, target_rank) * 0.1
            V_init = rng.randn(m*p, target_rank) * 0.1
            W_init = rng.randn(p*n, target_rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define objective function for this specific rank
            def objective(params):
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                size_W = p * n * target_rank
                
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:size_U+size_V+size_W]
                
                U = U_flat.reshape((n*m, target_rank))
                V = V_flat.reshape((m*p, target_rank))
                W = W_flat.reshape((p*n, target_rank))
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i*m + j]
                            v_val = V[j*p + k]
                            w_val = W[k*n + i]
                            s = np.dot(u_val, v_val * w_val)
                            error += (s - 1.0)**2
                return error

            try:
                res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                if res.success or res.fun < 1e-8:
                    U_opt = res.x[:n*m*target_rank].reshape((n*m, target_rank))
                    V_opt = res.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape((m*p, target_rank))
                    W_opt = res.x[n*m*target_rank+m*p*target_rank:].reshape((p*n, target_rank))
                    
                    valid, U_r, V_r, W_r = try_round_and_verify(U_opt, V_opt, W_opt, target_rank)
                    if valid:
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
            
    # Convert to list format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
