"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _compute_tensor(U, V, W, n, m, p):
    """Compute the decomposed tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T

def _error(T):
    """Compute max absolute error from all-ones tensor."""
    return np.max(np.abs(T - 1.0))

def _als_iteration(U, V, W, n, m, p):
    """
    One ALS iteration: solve for each factor given the other two.
    Returns new U, V, W.
    """
    rank = U.shape[1]
    U_new = np.zeros((n * m, rank))
    V_new = np.zeros((m * p, rank))
    W_new = np.zeros((p * n, rank))
    
    # Precompute V and W reshaped for faster access
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    U3 = U.reshape(n, m, rank)
    
    # Solve for U: for each (i,j), find U[i*m+j,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V3[j, k, :] * W3[k, i, :]
            b_vec = np.ones(p)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U_new[i * m + j, :] = sol
            except:
                U_new[i * m + j, :] = U[i * m + j, :]
    
    # Solve for V: for each (j,k), find V[j*p+k,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U3[i, j, :] * W3[k, i, :]
            b_vec = np.ones(n)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V_new[j * p + k, :] = sol
            except:
                V_new[j * p + k, :] = V[j * p + k, :]
    
    # Solve for W: for each (k,i), find W[k*n+i,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U3[i, j, :] * V3[j, k, :]
            b_vec = np.ones(m)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W_new[k * n + i, :] = sol
            except:
                W_new[k * n + i, :] = W[k * n + i, :]
    
    return U_new, V_new, W_new

def _als_with_polish(U, V, W, n, m, p, max_iter=80, tol=1e-12):
    """Run ALS iterations with polish."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor(U, V, W, n, m, p))
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        T = _compute_tensor(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Polish pass: tighter tolerance, more iterations
    if best_err < 0.1 and time.time() - _START_TIME < _TIMEOUT - 15:
        U, V, W = best_U, best_V, best_W
        for it in range(50):
            if time.time() - _START_TIME > _TIMEOUT - 5:
                break
            U, V, W = _als_iteration(U, V, W, n, m, p)
            T = _compute_tensor(U, V, W, n, m, p)
            err = _error(T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-14:
                break
    
    return best_U, best_V, best_W, best_err

def _generate_init(target_rank, n, m, p, restart):
    """Generate diverse initializations for a given rank."""
    np.random.seed(42 + restart * 1000)
    
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    U /= np.sqrt(target_rank)
    V /= np.sqrt(target_rank)
    W /= np.sqrt(target_rank)
    
    # Different initialization strategies based on restart
    if restart == 0:
        # Structured: partial schoolbook pattern
        for r in range(target_rank):
            i = r % n
            j = (r // n) % m
            k = (r // (n * m)) % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
            U[i * m + j, r] += 0.01 * np.random.randn()
            V[j * p + k, r] += 0.01 * np.random.randn()
            W[k * n + i, r] += 0.01 * np.random.randn()
    elif restart == 1:
        # Sparse random: each column has a few non-zero entries
        for r in range(target_rank):
            idx_u = np.random.choice(n * m, size=min(2, n * m), replace=False)
            idx_v = np.random.choice(m * p, size=min(2, m * p), replace=False)
            idx_w = np.random.choice(p * n, size=min(2, p * n), replace=False)
            U[idx_u, r] = np.random.randn(len(idx_u))
            V[idx_v, r] = np.random.randn(len(idx_v))
            W[idx_w, r] = np.random.randn(len(idx_w))
    elif restart == 2:
        # Block-diagonal inspired
        block_size = max(1, target_rank // 4)
        for r in range(target_rank):
            block = r // block_size
            i = block % n
            j = (block // n) % m
            k = (block // (n * m)) % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
            U[i * m + j, r] += 0.05 * np.random.randn()
            V[j * p + k, r] += 0.05 * np.random.randn()
            W[k * n + i, r] += 0.05 * np.random.randn()
    elif restart == 3:
        # Binary-like init
        U = (np.random.rand(n * m, target_rank) > 0.5).astype(float) * 2 - 1
        V = (np.random.rand(m * p, target_rank) > 0.5).astype(float) * 2 - 1
        W = (np.random.rand(p * n, target_rank) > 0.5).astype(float) * 2 - 1
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
    
    return U, V, W

def _try_rank(target_rank, n, m, p, num_restarts=8, time_budget=None):
    """Try to find a rank-target_rank decomposition."""
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    if time_budget is None:
        time_budget = 10  # seconds per rank
    
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _generate_init(target_rank, n, m, p, restart)
        U_sol, V_sol, W_sol, err = _als_with_polish(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
        
        # Early exit if we found exact solution
        if best_err < 1e-12:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = max_rank
    
    # Known lower bounds from literature for 3x4 * 4x4
    # Try ranks from low to high, but focus on likely candidates
    # The rank of matrix multiplication tensor is at least n*m + m*p + p*n - something
    # For 3,4,4: at least around 33-36
    
    # Strategy: try a range of ranks, spending more time on promising ones
    # Start from lower ranks and go up
    candidate_ranks = list(range(30, max_rank))
    
    for target_rank in candidate_ranks:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        
        # Spend more time on lower ranks (more likely to be minimal)
        if target_rank < 36:
            num_restarts = 10
        elif target_rank < 40:
            num_restarts = 8
        else:
            num_restarts = 5
        
        U_try, V_try, W_try, err = _try_rank(target_rank, n, m, p, num_restarts=num_restarts)
        
        if err < 1e-10:
            # Found exact decomposition
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = target_rank
            # Try even lower ranks
            continue
        else:
            # If error is small, try more restarts
            if err < 0.01 and time.time() - _START_TIME < _TIMEOUT - 15:
                U_try2, V_try2, W_try2, err2 = _try_rank(target_rank, n, m, p, num_restarts=10)
                if err2 < 1e-10:
                    best_U, best_V, best_W = U_try2, V_try2, W_try2
                    best_rank = target_rank
                    continue
    
    # Convert to lists for return
    U_list = [[float(U_list_val) for U_list_val in row] for row in best_U.tolist()]
    V_list = [[float(V_list_val) for V_list_val in row] for row in best_V.tolist()]
    W_list = [[float(W_list_val) for W_list_val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
