"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def _rank_of_matrix(M, tol=1e-10):
    """Compute numerical rank of matrix M."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * (s[0] if len(s) > 0 else 1.0)))

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=500, tol=1e-10):
    """
    Try to reduce rank by optimizing with fixed target_rank.
    Uses a penalty-based approach to encourage sparsity.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        # Already at or below target rank, just verify
        err = _check_decomposition(U, V, W, n, m, p)
        if err < tol:
            return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank], True
        return None, None, None, False
    
    # We need to find a rank-target_rank decomposition
    # Parameterize all three matrices
    U_init = np.random.randn(n * m, target_rank) * 0.3
    V_init = np.random.randn(m * p, target_rank) * 0.3
    W_init = np.random.randn(p * n, target_rank) * 0.3
    
    def objective_and_grad(x):
        U_flat = x[:n*m*target_rank]
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = x[n*m*target_rank + m*p*target_rank :]
        
        U_mat = U_flat.reshape(n * m, target_rank)
        V_mat = V_flat.reshape(m * p, target_rank)
        W_mat = W_flat.reshape(p * n, target_rank)
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    obj += (val - 1.0) ** 2
        
        return obj
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if result.fun < 1e-8:
            U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
            V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
            W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
            
            # Verify
            err = _check_decomposition(U_found, V_found, W_found, n, m, p)
            if err < 1e-6:
                return U_found, V_found, W_found, True
    except Exception:
        pass
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook is valid
    err = _check_decomposition(U, V, W, n, m, p)
    if err < 1e-10:
        U_int = [[int(round(x)) for x in row] for row in U]
        V_int = [[int(round(x)) for x in row] for row in V]
        W_int = [[int(round(x)) for x in row] for row in W]
    
    # For 3x3x6, known bounds suggest rank can be significantly less than 54
    # Try ranks from 54 down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But actual lower bound for tensor rank is higher
    # Try a range of ranks
    
    rank_targets = list(range(54, 17, -1))
    
    for target_rank in rank_targets:
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(8):
            if time.time() - start_time > deadline - 10:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 31)
            
            U_r, V_r, W_r, success = _try_reduce_rank(
                U, V, W, n, m, p, target_rank, max_iter=300
            )
            
            if success:
                # Try to round to half-integers
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
        
        # If we found a valid decomposition at this rank, stop
        if found:
            # Try to go even lower
            continue
    
    # Final result
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
