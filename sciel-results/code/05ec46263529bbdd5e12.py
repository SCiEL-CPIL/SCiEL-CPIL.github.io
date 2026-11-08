"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _build_tensor_constraints(n, m, p):
    """Return list of (U_idx, V_idx, W_idx) for each tensor entry that must equal 1."""
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    return constraints

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    rounded = np.round(x * 2) / 2.0
    if np.max(np.abs(x - rounded)) < tol:
        return rounded
    return x

def _objective(x, n, m, p, rank):
    """Compute the sum of squared errors for a given parameter vector."""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U_mat = U_flat.reshape(n * m, rank)
    V_mat = V_flat.reshape(m * p, rank)
    W_mat = W_flat.reshape(p * n, rank)
    
    err_sum = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                err_sum += (val - 1.0) ** 2
    return err_sum

def _structured_init(n, m, p, rank, seed=0):
    """
    Create a structured initialization that respects the algebraic structure.
    For 3x3x6, we split the 6 columns into two groups of 3.
    """
    np.random.seed(seed)
    
    # If rank is even and p=6, use block structure for 2x (3x3x3)
    if p == 6 and rank % 2 == 0:
        half_rank = rank // 2
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Block 1: k=0..2
        U[:, :half_rank] = np.random.randn(n * m, half_rank) * 0.5
        V[:m*3, :half_rank] = np.random.randn(m * 3, half_rank) * 0.5
        W[:3*n, :half_rank] = np.random.randn(3 * n, half_rank) * 0.5
        
        # Block 2: k=3..5
        U[:, half_rank:] = np.random.randn(n * m, rank - half_rank) * 0.5
        V[m*3:, half_rank:] = np.random.randn(m * 3, rank - half_rank) * 0.5
        W[3*n:, half_rank:] = np.random.randn(3 * n, rank - half_rank) * 0.5
        
        return U, V, W
    
    # Fallback random init
    U = np.random.randn(n * m, rank) * 0.3
    V = np.random.randn(m * p, rank) * 0.3
    W = np.random.randn(p * n, rank) * 0.3
    return U, V, W

def _try_find_decomposition(n, m, p, target_rank, max_time=20.0):
    """
    Try to find a rank-target_rank decomposition using local optimization
    with multiple structured initializations.
    """
    start_time = time.time()
    
    def local_optimize(x0, max_iter=500):
        try:
            result = minimize(_objective, x0, args=(n, m, p, target_rank),
                            method='L-BFGS-B', 
                            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
            return result
        except Exception:
            return None
    
    # Try multiple structured initializations with different seeds
    for seed in range(20):
        if time.time() - start_time > max_time:
            break
            
        U_init, V_init, W_init = _structured_init(n, m, p, target_rank, seed=seed)
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        result = local_optimize(x0, max_iter=300)
        if result is not None and result.fun < 1e-10:
            U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
            V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
            W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
            
            err = _check_decomposition(U_found, V_found, W_found, n, m, p)
            if err < 1e-8:
                return U_found, V_found, W_found
    
    return None, None, None

def _high_precision_polish(U, V, W, n, m, p, max_iter=2000):
    """
    High-precision polish step to tighten the solution.
    """
    rank = U.shape[1]
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        return _objective(x, n, m, p, rank)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if result.fun < 1e-12:
            U_polished = result.x[:n*m*rank].reshape(n * m, rank)
            V_polished = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
            W_polished = result.x[n*m*rank + m*p*rank :].reshape(p * n, rank)
            
            err = _check_decomposition(U_polished, V_polished, W_polished, n, m, p)
            if err < 1e-10:
                return U_polished, V_polished, W_polished
    except Exception:
        pass
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    err = _check_decomposition(U, V, W, n, m, p)
    if err >= 1e-10:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Try to find lower rank decompositions
    # Prioritize ranks that are multiples of small numbers (structural hints)
    target_ranks = [46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    for target_rank in target_ranks:
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        # Try multiple seeds for lower ranks
        max_seeds = 10 if target_rank <= 46 else 5
        for seed in range(max_seeds):
            if time.time() - start_time > deadline - 10:
                break
            
            np.random.seed(seed * 7919 + target_rank * 31)
            
            U_r, V_r, W_r = _try_find_decomposition(n, m, p, target_rank, max_time=15.0)
            
            if U_r is not None:
                U_r, V_r, W_r = _high_precision_polish(U_r, V_r, W_r, n, m, p)
                
                U_rounded = _to_half_int(U_r)
                V_rounded = _to_half_int(V_r)
                W_rounded = _to_half_int(W_r)
                
                if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                else:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                
                found = True
                break
        
        if found:
            break
    
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
