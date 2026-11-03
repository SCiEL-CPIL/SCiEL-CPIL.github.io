"""Matrix multiplication 3x3x6 tensor decomposition - search for minimal rank."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _build_target_tensor():
    """Build the target multiplication tensor T[i,j,k] = 1 for all valid indices."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the multiplication tensor."""
    rank = U.shape[1]
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                val = np.dot(u_row * v_row, w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_step(U, V, W, n, m, p, max_iter=50, tol=1e-8):
    """Alternating Least Squares refinement of the factorization."""
    rank = U.shape[1]
    for it in range(max_iter):
        # Update U: for each row (i*m+j), solve least squares
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is: U_row @ (V_row * W_row) = 1 for each k
                # Stack: for each k, the row is V[j*p+k,:] * W[k*n+i,:]
                # So U_row @ M = ones(p) where M[k,:] = V[j*p+k,:] * W[k*n+i,:]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve U_row @ M^T = ones(p), i.e., M @ U_row^T = ones(p)
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = sol
                except np.linalg.LinAlgError:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = sol
                except np.linalg.LinAlgError:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = sol
                except np.linalg.LinAlgError:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        # Check convergence
        change = np.max(np.abs(U_new - U) + np.abs(V_new - V) + np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        if change < tol:
            break
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the maximum error in the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank(rank, n, m, p, n_restarts=20, max_als_iter=30, deadline=None):
    """Try to find a valid decomposition of the given rank."""
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline - 2.0:
            break
        
        # Initialize with random factors
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Run ALS
        U, V, W = _als_step(U, V, W, n, m, p, max_iter=max_als_iter)
        
        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-6:
            return U, V, W
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None, None, None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find lower ranks
    # Theoretical lower bounds: for 3x3x6, known results suggest rank around 40-50
    # Try from rank 40 up to 53
    for rank in range(40, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        n_restarts = max(5, int((deadline - 5.0 - time.time()) / 3.0))
        n_restarts = min(n_restarts, 30)
        
        U, V, W = _try_rank(rank, n, m, p, n_restarts=n_restarts, 
                           max_als_iter=40, deadline=deadline)
        
        if U is not None:
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break  # Found a valid decomposition at this rank
    
    return best_result
