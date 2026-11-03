import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to nearest 0.5."""
    return np.round(M * 2.0) / 2.0

def _tensor_residuals(params, R, n, m, p):
    """
    Compute residuals for the tensor decomposition.
    params: flattened array of U, V, W.
    """
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    U = params[:size_U].reshape(n * m, R)
    V = params[size_U:size_U + size_V].reshape(m * p, R)
    W = params[size_U + size_V:].reshape(p * n, R)
    
    residuals = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                residuals.append(1.0 - val)
    
    return np.array(residuals)

def _try_rank(R, n, m, p, time_limit, seed_base=0):
    """
    Try to find a decomposition of rank R.
    """
    start_time = time.time()
    max_attempts = 10
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        seed = seed_base + attempt * 137
        np.random.seed(seed)
        
        scale = 1.0 / np.sqrt(R)
        
        U_init = np.random.randn(n * m, R) * scale
        V_init = np.random.randn(m * p, R) * scale
        W_init = np.random.randn(p * n, R) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(
                _tensor_residuals, 
                params_init, 
                args=(R, n, m, p),
                method='trf', 
                max_nfev=500
            )
        except Exception:
            continue
            
        if result.cost < 1e-8:
            params_sol = result.x
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            
            U_sol = params_sol[:size_U].reshape(n * m, R)
            V_sol = params_sol[size_U:size_U + size_V].reshape(m * p, R)
            W_sol = params_sol[size_U + size_V:].reshape(p * n, R)
            
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                U_snap = _snap_to_half_integer(U_sol)
                V_snap = _snap_to_half_integer(V_sol)
                W_snap = _snap_to_half_integer(W_sol)
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                else:
                    # Accept if entries are already close to half-integers
                    all_close = True
                    for mat in [U_sol, V_sol, W_sol]:
                        for val in mat.flatten():
                            if abs(val * 2 - round(val * 2)) > 1e-4:
                                all_close = False
                                break
                        if not all_close:
                            break
                    if all_close and _verify(U_sol, V_sol, W_sol, n, m, p, tol=1e-10):
                        return U_sol, V_sol, W_sol
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s safety margin
    
    # Schoolbook fallback (rank 96) - guaranteed valid
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Search for lower rank
    for R in range(20, 96):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 2:
            break
        
        time_per_rank = min(1.5, time_left / max(1, 96 - R))
        
        result = _try_rank(R, n, m, p, time_limit=time_per_rank, seed_base=R)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_result = {
                "U": U_sol.tolist(),
                "V": V_sol.tolist(),
                "W": W_sol.tolist()
            }
            return best_result
            
    return best_result
