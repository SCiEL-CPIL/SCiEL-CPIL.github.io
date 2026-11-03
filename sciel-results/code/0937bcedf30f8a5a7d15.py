"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    # Vectorized verification
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We can check the max absolute difference from 1
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (s - 1.0)**2
    return err < tol

def _tensor_residual(params, n, m, p, rank):
    """Calculate the squared Frobenius norm of the residual tensor."""
    # Unpack params
    U_flat = params[:n*m*rank]
    V_flat = params[n*m*rank : n*m*rank + m*p*rank]
    W_flat = params[n*m*rank + m*p*rank : ]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    err = 0.0
    # Iterate over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # s = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (s - 1.0)**2
    return err

def _try_rank_lbfgs(n, m, p, rank, seed=0, deadline=None):
    """Try to find a rank-r decomposition using L-BFGS-B."""
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    # Flatten
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _tensor_residual(x, n, m, p, rank)
    
    # L-BFGS-B is good for this size
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
    
    if res.fun < 1e-6:
        # Reconstruct
        U_flat = res.x[:n*m*rank]
        V_flat = res.x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = res.x[n*m*rank + m*p*rank : ]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Verify with tighter tol
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    school_rank = n * m * p
    
    # Fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, school_rank
    
    # Try ranks from lower bound to schoolbook - 1
    # Lower bound for 3x3x6 is likely around 20-30.
    # We know 46 is possible (2x23).
    # Let's try a range.
    ranks_to_try = list(range(20, school_rank))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
        found = False
        # Fewer seeds for L-BFGS as it's more robust but slower per run
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
            U, V = _try_rank_lbfgs(n, m, p, rank, seed=seed, deadline=deadline)
            if U is not None:
                best_U, best_V, best_W, best_rank = U, V, W, rank
                found = True
                break
        if found:
            break
            
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
