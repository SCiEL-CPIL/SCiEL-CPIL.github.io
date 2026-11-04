"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product index."""
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
    
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = recon - T_flat.reshape(n * m, m * p, p * n)
    return np.sum(err * err)

def _try_rank(n, m, p, rank, T_flat, time_budget):
    """Try to find a rank-rank decomposition with given rank."""
    start_time = time.time()
    best_err = np.inf
    best_params = None
    
    n_params = (n * m + m * p + p * n) * rank
    max_restarts = 200 # High number, but limited by time
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget * 0.95:
            break
        
        # Use a deterministic seed based on restart for reproducibility in this run
        seed = restart * 12345 + 67890
        rng = np.random.default_rng(seed)
        
        # Initialization: Small random noise
        params_init = rng.normal(0, 1.0, n_params)
        
        try:
            result = minimize(
                _reconstruction_error,
                params_init,
                args=(n, m, p, rank, T_flat),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            err = _reconstruction_error(result.x, n, m, p, rank, T_flat)
            if err < best_err:
                best_err = err
                best_params = result.x.copy()
                
            # Early exit if we found a near-exact solution
            if err < 1e-8:
                break
                
        except Exception:
            continue
            
    return best_err, best_params

def _snap_to_rational(params, n, m, p, rank, tol=1e-6):
    """Try to snap parameters to simple rational values (integers or halves)."""
    U = params[:n * m * rank].reshape(n * m, rank)
    V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
    W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
    
    T = _build_tensor(n, m, p)
    
    # Try snapping to multiples of 0.5
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
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
        
    # Try snapping to multiples of 0.25? Unlikely but cheap to check
    U_snap = np.round(U * 4) / 4
    V_snap = np.round(V * 4) / 4
    W_snap = np.round(W * 4) / 4
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    err = np.sum((recon - T) ** 2)
    if err < tol:
        return U_snap, V_snap, W_snap

    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    total_start = time.time()
    time_budget = 110.0
    
    # Fallback: Schoolbook rank 24
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 19 up to 23
    for rank in range(19, min(24, rank_fb)):
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time: Rank 19 gets the most
        if rank == 19:
            rank_budget = remaining * 0.6
        elif rank == 20:
            rank_budget = remaining * 0.2
        else:
            rank_budget = remaining * 0.06
            
        if rank_budget < 1.0:
            continue

        best_err, best_params = _try_rank(n, m, p, rank, T_flat, rank_budget)
        
        if best_err < 1e-8 and best_params is not None:
            snapped = _snap_to_rational(best_params, n, m, p, rank)
            if snapped is not None:
                U_snap, V_snap, W_snap = snapped
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            # If snapping fails but error is tiny, return floats (might pass if tolerance is loose, but spec says integer/half-integer)
            # The spec says "with integer/half-integer entries". If we can't snap, it's invalid.
            # So we must only return if snapped is not None.
            
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
