"""Tensor decomposition search for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err

def _als_decompose(n, m, p, target_rank, max_iters=200, tol=1e-8, seed=42):
    """Alternating Least Squares to find rank-r decomposition."""
    np.random.seed(seed)
    
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    for it in range(max_iters):
        # Update U: for each row (i,j), solve linear system
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # Let M_u[c, k] = V[j*p+k, c] * W[k*n+i, c]
        # Then U[i*m+j, :] @ M_u = 1 (vector of ones, length p)
        
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                # Build matrix M_u of shape (target_rank, p)
                M_u = np.zeros((target_rank, p))
                for k in range(p):
                    M_u[:, k] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j, :] @ M_u = ones(p)
                # This is: M_u.T @ U[i*m+j, :] = ones(p)
                try:
                    sol = np.linalg.lstsq(M_u.T, np.ones(p), rcond=None)[0]
                    U_new[i*m+j, :] = sol
                except:
                    pass
        U = U_new
        
        # Update V: for each row (j,k), solve linear system
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                M_v = np.zeros((target_rank, n))
                for i in range(n):
                    M_v[:, i] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol = np.linalg.lstsq(M_v.T, np.ones(n), rcond=None)[0]
                    V_new[j*p+k, :] = sol
                except:
                    pass
        V = V_new
        
        # Update W: for each row (k*n+i), solve linear system
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        W_new = W.copy()
        for i in range(n):
            for k in range(p):
                M_w = np.zeros((target_rank, m))
                for j in range(m):
                    M_w[:, j] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol = np.linalg.lstsq(M_w.T, np.ones(m), rcond=None)[0]
                    W_new[k*n+i, :] = sol
                except:
                    pass
        W = W_new
        
        # Check convergence
        valid, max_err = _validate(U, V, W, n, m, p, tol=1e-4)
        if max_err < tol:
            break
    
    valid, max_err = _validate(U, V, W, n, m, p)
    if valid:
        return U, V, W, max_err
    return None, None, None, max_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=np.float64)
    V_school = np.zeros((m*p, rank_school), dtype=np.float64)
    W_school = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank decompositions
    # Try various target ranks from low to high
    candidate_ranks = list(range(16, rank_school + 1, 4))
    candidate_ranks.sort()  # Try lower ranks first
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W, max_err = _als_decompose(n, m, p, target_rank, max_iters=100, seed=seed)
            
            if U is not None and max_err < 1e-6:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break  # Found valid decomposition at this rank, move to next lower rank
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
