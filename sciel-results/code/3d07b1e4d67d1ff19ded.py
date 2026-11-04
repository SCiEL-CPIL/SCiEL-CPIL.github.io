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

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = np.sum((recon - T) ** 2)
    return err < tol

def _snap_matrix(M, tol=1e-6):
    """Snap matrix entries to nearest integer or half-integer."""
    M_int = np.round(M)
    if np.allclose(M, M_int, atol=tol):
        return M_int
    
    M_half = np.round(M * 2) / 2.0
    if np.allclose(M, M_half, atol=tol):
        return M_half
        
    return None

def _reconstruction_error(params, n, m, p, rank, T):
    """Calculate the squared Frobenius norm error of the decomposition."""
    # params is a flat vector of size rank * (n*m + m*p + p*n)
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_flat = params[:size_u]
    V_flat = params[size_u:size_u + size_v]
    W_flat = params[size_u + size_v:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = np.sum((recon - T) ** 2)
    return err

def _search_rank_opt(n, m, p, rank, T, time_budget):
    """
    Search for a rank-rank decomposition using gradient-based optimization.
    """
    start_time = time.time()
    
    # Initial random guesses
    np.random.seed(42)
    
    # We perform multiple restarts
    max_restarts = 50
    time_per_restart = time_budget / max_restarts
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Random initialization
        U0 = np.random.randn(n * m, rank) * 0.5
        V0 = np.random.randn(m * p, rank) * 0.5
        W0 = np.random.randn(p * n, rank) * 0.5
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Define objective
        def objective(x):
            return _reconstruction_error(x, n, m, p, rank, T)
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
            
        if res.success or res.fun < 1e-6:
            # Extract optimized matrices
            x_opt = res.x
            size_u = n * m * rank
            size_v = m * p * rank
            size_w = p * n * rank
            
            U_opt = x_opt[:size_u].reshape(n * m, rank)
            V_opt = x_opt[size_u:size_u + size_v].reshape(m * p, rank)
            W_opt = x_opt[size_u + size_v:].reshape(p * n, rank)
            
            # Try to snap to integers/halves
            U_snap = _snap_matrix(U_opt)
            V_snap = _snap_matrix(V_opt)
            W_snap = _snap_matrix(W_opt)
            
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if _check_decomposition(U_snap, V_snap, W_snap, T):
                    return U_snap, V_snap, W_snap
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Deadline guard: must finish within 150s, use 140s budget
    total_start = time.time()
    time_budget = 140.0
    
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
    for rank in [19, 20, 21, 22, 23]:
        if rank > rank_fb:
            continue
            
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time: give most to 19
        if rank == 19:
            rank_budget = remaining * 0.8
        else:
            rank_budget = remaining * 0.1
            
        result = _search_rank_opt(n, m, p, rank, T, rank_budget)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
