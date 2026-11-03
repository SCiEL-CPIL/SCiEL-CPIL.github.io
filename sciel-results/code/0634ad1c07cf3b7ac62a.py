import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, leave 20s margin
    
    # Precompute indices
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    # Target: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    
    def residuals(x, rank):
        # x is flattened U, V, W
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i,j,:] * V[j,k,:] * W[k,i,:]
                    res[i, j, k] = 1.0 - np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
        return res.flatten()
    
    def try_rank(rank):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W
        
        # Initialize
        # Random initialization
        np.random.seed(42) # Fixed seed for reproducibility in search, but we might want variation
        # Actually, let's try a few random seeds
        
        best_solution = None
        best_err = float('inf')
        
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart)
            x0 = np.random.randn(total_size)
            
            try:
                result = least_squares(residuals, x0, args=(rank,), method='lm', max_nfev=200)
                err = np.sum(result.fun**2)
                
                if err < best_err:
                    best_err = err
                    best_solution = result.x
                    
                # If error is very small, try to round and verify
                if err < 1e-6:
                    # Unpack
                    U_flat = best_solution[:size_U]
                    V_flat = best_solution[size_U:size_U + size_V]
                    W_flat = best_solution[size_U + size_V:]
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_flat * 2) / 2
                    V_rounded = np.round(V_flat * 2) / 2
                    W_rounded = np.round(W_flat * 2) / 2
                    
                    # Verify
                    U_r = U_rounded.reshape(n, m, rank)
                    V_r = V_rounded.reshape(m, p, rank)
                    W_r = W_rounded.reshape(p, n, rank)
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        # Construct return format
                        U_ret = U_rounded.reshape(n * m, rank).tolist()
                        V_ret = V_rounded.reshape(m * p, rank).tolist()
                        W_ret = W_rounded.reshape(p * n, rank).tolist()
                        return {"U": U_ret, "V": V_ret, "W": W_ret}
                        
            except Exception:
                pass
                
        return None

    # Try ranks from 26 up to 35
    for rank in range(26, rank_schoolbook):
        if time.time() > deadline:
            break
        sol = try_rank(rank)
        if sol is not None:
            return sol
            
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
