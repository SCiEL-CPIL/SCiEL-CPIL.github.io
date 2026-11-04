"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product index."""
    # T has shape (n*m, m*p, p*n) where T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _reconstruction_error(params, n, m, p, rank, T_flat):
    """Compute squared Frobenius norm of reconstruction error."""
    U = params[:n * m * rank].reshape(n * m, rank)
    V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
    W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
    
    # Compute reconstruction: sum_c U[:,c] * V[:,c] * W[:,c]
    # Reshape to tensor form
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = recon - T_flat.reshape(n * m, m * p, p * n)
    return np.sum(err * err)

def _try_rank(n, m, p, rank, T_flat, time_budget):
    """Try to find a rank-rank decomposition with given rank."""
    start_time = time.time()
    best_err = np.inf
    best_params = None
    
    n_params = (n * m + m * p + p * n) * rank
    max_restarts = max(1, int(time_budget / 0.5))  # ~0.5s per restart
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget * 0.9:
            break
        
        # Random initialization with small values
        np.random.seed(hash((n, m, p, rank, restart)) % (2**32))
        params = np.random.randn(n_params) * 0.1
        
        # Try a few optimization steps with different initializations
        for init_scale in [0.1, 0.5, 1.0, 2.0]:
            if time.time() - start_time > time_budget * 0.9:
                break
            
            params_init = np.random.randn(n_params) * init_scale
            
            try:
                result = minimize(
                    _reconstruction_error,
                    params_init,
                    args=(n, m, p, rank, T_flat),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 500}
                )
                
                err = _reconstruction_error(result.x, n, m, p, rank, T_flat)
                if err < best_err:
                    best_err = err
                    best_params = result.x.copy()
                    
                # Also check if we can polish with higher precision
                if err < 1e-8:
                    # Very close to zero - try to snap to rational values
                    break
                    
            except Exception:
                continue
    
    return best_err, best_params

def _snap_to_rational(params, n, m, p, rank, tol=1e-6):
    """Try to snap parameters to simple rational values (integers or halves)."""
    U = params[:n * m * rank].reshape(n * m, rank)
    V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
    W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
    
    # Try snapping to multiples of 0.5
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Check if snapped version still works
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    T = _build_tensor(n, m, p)
    err = np.sum((recon - T) ** 2)
    
    if err < tol:
        return U_snap, V_snap, W_snap
    
    # Try snapping to integers
    U_snap = np.round(U)
    V_snap = np.round(V)
    W_snap = np.round(W)
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    err = np.sum((recon - T) ** 2)
    
    if err < tol:
        return U_snap, V_snap, W_snap
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Deadline guard: must finish within 120s, use 110s budget
    total_start = time.time()
    time_budget = 110.0
    
    # Start with schoolbook rank 24 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try ranks from 19 up to 23
    for rank in range(19, min(24, rank_fb)):
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time proportionally, give more to lower ranks
        rank_budget = remaining * 0.3 if rank == 19 else remaining * 0.2
        
        best_err, best_params = _try_rank(n, m, p, rank, T_flat, rank_budget)
        
        if best_err < 1e-8 and best_params is not None:
            # Found a valid decomposition, try to snap to rational
            snapped = _snap_to_rational(best_params, n, m, p, rank)
            if snapped is not None:
                U_snap, V_snap, W_snap = snapped
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            # Return the float version (should be close enough)
            U = best_params[:n * m * rank].reshape(n * m, rank)
            V = best_params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
            W = best_params[n * m * rank + m * p * rank:].reshape(p * n, rank)
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
