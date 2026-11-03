"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return err

def _alternating_minimize(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """Alternating minimization for tensor decomposition."""
    for it in range(max_iter):
        # Update U: for each (i,j), solve linear system
        # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is: U[i*m+j,:] @ M_ij = 1 for all k, where M_ij[k,c] = V[j*p+k,c]*W[k*n+i,c]
        # Actually: for fixed i,j, we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # Let A_ij be a p x r matrix with A_ij[k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then we need U[i*m+j,:] @ A_ij.T = ones(p)
        # So U[i*m+j,:] = ones(p) @ A_ij.T^{-1} (if invertible) or least squares
        
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                # Build A_ij: p x r
                A_ij = np.zeros((p, U.shape[1]))
                for k in range(p):
                    A_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve: U[i*m+j,:] @ A_ij.T = ones(p)
                # i.e., A_ij @ U[i*m+j,:].T = ones(p)
                try:
                    sol = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)[0]
                    U_new[i*m+j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k), solve
        # sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                A_jk = np.zeros((n, V.shape[1]))
                for i in range(n):
                    A_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)[0]
                    V_new[j*p+k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i), solve
        # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                A_ki = np.zeros((m, W.shape[1]))
                for j in range(m):
                    A_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)[0]
                    W_new[k*n+i, :] = sol
                except:
                    pass
        
        U, V, W = U_new, V_new, W_new
        err = _compute_error(U, V, W, n, m, p)
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank(target_rank, n, m, p, n_attempts=10, max_iter=300, deadline=None):
    """Try to find a decomposition of given rank."""
    for attempt in range(n_attempts):
        if deadline and time.time() > deadline:
            return None
        
        # Random initialization with some structure
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        U, V, W, err = _alternating_minimize(U, V, W, n, m, p, max_iter=max_iter)
        
        if err < 1e-8:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook decomposition as fallback
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s safety margin
    
    best_rank = rank_school
    best_U, best_V, best_W = U0, V0, W0
    
    # Try ranks from 74 down to 50
    for target_rank in range(rank_school - 1, 49, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank(target_rank, n, m, p, n_attempts=5, max_iter=200, deadline=deadline)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
