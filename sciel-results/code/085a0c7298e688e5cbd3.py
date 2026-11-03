"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_sub_decomposition(n, m, p, R, max_iter=1000, trials=10):
    """Find a rank-R decomposition for n x m x p matrix multiplication."""
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(trials):
        np.random.seed(42 + trial)
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        def flatten_params():
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unpack_params(x):
            U_flat = x[:n*m*R]
            V_flat = x[n*m*R:n*m*R + m*p*R]
            W_flat = x[n*m*R + m*p*R:]
            return U_flat.reshape(n*m, R), V_flat.reshape(m*p, R), W_flat.reshape(p*n, R)
        
        def objective(x):
            U_opt, V_opt, W_opt = unpack_params(x)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_opt[i*m+j, :] * V_opt[j*p+k, :] * W_opt[k*n+i, :])
                        err += (val - 1.0) ** 2
            return err
        
        x0 = flatten_params()
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e-8:
                U_opt, V_opt, W_opt = unpack_params(result.x)
                
                # Round to nearest integer or half-integer
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                if _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    return U_rounded, V_rounded, W_rounded
                    
        except Exception:
            continue
            
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition using sub-problem structure
    # p=9, so we can split into 3 blocks of p=3
    if p % 3 == 0:
        p_sub = p // 3
        # Try ranks from 11 to 18 for the sub-problem
        for R_sub in range(11, 19):
            if time.time() > deadline - 20.0:
                break
            
            U_sub, V_sub, W_sub = _find_sub_decomposition(n, m, p_sub, R_sub)
            
            if U_sub is not None:
                # Construct full decomposition
                R_total = R_sub * 3
                U_full = np.zeros((n*m, R_total))
                V_full = np.zeros((m*p, R_total))
                W_full = np.zeros((p*n, R_total))
                
                for block in range(3):
                    r_offset = block * R_sub
                    # U is same for all blocks
                    U_full[:, r_offset:r_offset+R_sub] = U_sub
                    
                    # V: map k in full (0..8) to k' in sub (0..2)
                    for j in range(m):
                        for k in range(p):
                            k_sub = k % p_sub
                            V_full[j*p + k, r_offset:r_offset+R_sub] = V_sub[j*p_sub + k_sub, :]
                            
                    # W: map k in full (0..8) to k' in sub (0..2)
                    for i in range(n):
                        for k in range(p):
                            k_sub = k % p_sub
                            W_full[k*n + i, r_offset:r_offset+R_sub] = W_sub[k_sub*n + i, :]
                
                # Verify full decomposition
                if _verify(U_full, V_full, W_full, n, m, p, tol=1e-6):
                    best_U = U_full
                    best_V = V_full
                    best_W = W_full
                    best_rank = R_total
                    break # Found a valid decomposition, stop searching for lower ranks
            
    U_out = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_out = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_out = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
