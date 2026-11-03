"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import random
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(W.shape[1]):
                w_val = W[k * n + i, r]
                if w_val == 0: continue
                u_vec = U[i * m : (i + 1) * m, r]
                v_vec = np.array([V[j * p + k, r] for j in range(m)])
                M += w_val * np.outer(u_vec, v_vec)
            if not np.allclose(M, np.eye(m), atol=tol):
                return False
    return True

def _snap_to_half_int(arr):
    # Try to snap to nearest half-integer
    snapped = np.round(arr * 2) / 2
    return snapped

def _try_rank(n, m, p, target_rank, time_limit=5):
    start_time = time.time()
    # Initialize with random values
    # The system is: sum_r U[i,j,r] V[l,k,r] W[k,i,r] = delta[j,l]
    # Variables: U (6*R), V (12*R), W (8*R)
    # Total 26*R variables.
    
    # We can fix the scaling of one matrix to remove the scaling ambiguity?
    # Or just use random initialization.
    
    # Let's try a few random restarts
    best_residual = float('inf')
    best_params = None
    
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale matters, let's use small random values
        U_flat = np.random.randn(n * m * target_rank) * 0.1
        V_flat = np.random.randn(m * p * target_rank) * 0.1
        W_flat = np.random.randn(p * n * target_rank) * 0.1
        
        def residuals(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            res = []
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for l in range(m):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m + j, r] * V[l*p + k, r] * W[k*n + i, r]
                            target = 1.0 if j == l else 0.0
                            res.append(val - target)
            return np.array(res)
        
        # Use least squares
        # This might be slow if R is large, but R is small here.
        # 72 residuals, 26*R variables.
        # For R=23, 598 variables.
        
        try:
            res = opt.least_squares(residuals, np.concatenate([U_flat, V_flat, W_flat]), method='lm', max_nfev=1000)
            if res.cost < best_residual:
                best_residual = res.cost
                best_params = res.x
        except Exception:
            continue
            
    if best_params is not None and best_residual < 1e-8:
        U = best_params[:n*m*target_rank].reshape(n*m, target_rank)
        V = best_params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = best_params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Snap to half integers
        U_snapped = _snap_to_half_int(U)
        V_snapped = _snap_to_half_int(V)
        W_snapped = _snap_to_half_int(W)
        
        if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
            return U_snapped, V_snapped, W_snapped
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 120.0
    start_time = time.time()
    
    # Try to find a lower rank
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try ranks from schoolbook - 1 down to some limit
    # We only have 120s.
    # Let's try 23, 22, 21.
    
    for r in range(best_rank - 1, max(10, best_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        U, V, W = _try_rank(n, m, p, r, time_limit=10)
        if U is not None:
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break # Found a better one, stop? Or continue to find even better?
            # Given the difficulty, finding one is a win.
            
    if best_U is None:
        best_U, best_V, best_W = _schoolbook(n, m, p)
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
