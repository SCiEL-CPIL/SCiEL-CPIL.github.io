"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from scipy.optimize import minimize

def _make_schoolbook(n, m, p):
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

def _to_lists(U, V, W):
    def snap(x):
        # Snap to nearest half-integer
        return round(x * 2) / 2.0
    
    U_list = [[snap(x) for x in row] for row in U.tolist()]
    V_list = [[snap(x) for x in row] for row in V.tolist()]
    W_list = [[snap(x) for x in row] for row in W.tolist()]
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }

def _verify(U, V, W, n, m, p, tol=1e-8):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W is indexed by k, i in the tensor definition W[k*n+i, r]
    # So W_t[k, i, r]
    
    # Use einsum for speed
    # U_t[i, j, r]
    # V_t[j, k, r]
    # W_t[k, i, r] -> We need to align indices.
    # Let's transpose W_t to (i, k, r) to match i, j, k order?
    # W_t is (p, n, rank) which is (k, i, r).
    # We want T[i, j, k].
    # Term is U[i, j, r] * V[j, k, r] * W[k, i, r].
    # einsum: 'ijr,jkr,kir->ijk'
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception:
        return False, 1.0
        
    error = np.max(np.abs(T - 1.0))
    return error < tol, error

def _compute_error(U_flat, V_flat, W_flat, n, m, p, rank):
    """Compute max absolute error given flattened parameters."""
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = T - 1.0
    return np.max(np.abs(err))

def _objective(params, n, m, p, rank):
    """Smooth surrogate objective using log-sum-exp."""
    U_flat = params[:n*m*rank]
    V_flat = params[n*m*rank : n*m*rank + m*p*rank]
    W_flat = params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = np.abs(T - 1.0)
    
    # Use log-sum-exp for smooth max
    max_err = np.max(err)
    if max_err < 1e-12:
        return 0.0
    
    # Soft max: exp(beta * err) sum, then log / beta
    beta = 10.0
    shifted = beta * err
    shifted_max = np.max(shifted)
    log_sum_exp = shifted_max + np.log(np.sum(np.exp(shifted - shifted_max)))
    soft_max = log_sum_exp / beta
    
    # Add small regularization to encourage sparse/simple solutions
    reg = 1e-6 * (np.sum(U_flat**2) + np.sum(V_flat**2) + np.sum(W_flat**2))
    
    return soft_max + reg

def _try_optimize(U_init, V_init, W_init, n, m, p, rank, deadline, max_time_per_attempt=5.0):
    """Run scipy optimization from given initialization."""
    U_flat = U_init.flatten()
    V_flat = V_init.flatten()
    W_flat = W_init.flatten()
    
    params = np.concatenate([U_flat, V_flat, W_flat])
    
    start_time = time.time()
    timeout = min(deadline - start_time, max_time_per_attempt)
    if timeout < 0.5:
        return None, None, None, 1.0
    
    try:
        result = minimize(
            _objective,
            params,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > timeout:
            return None, None, W_init, 1.0
        
        final_params = result.x
        U_opt = final_params[:n*m*rank].reshape(n * m, rank)
        V_opt = final_params[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
        W_opt = final_params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p * n, rank)
        
        err = _compute_error(U_opt.flatten(), V_opt.flatten(), W_opt.flatten(), n, m, p, rank)
        
        return U_opt, V_opt, W_opt, err
        
    except Exception as e:
        return None, None, None, 1.0

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = _to_lists(U0, V0, W0)
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            scale = 1.0
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Run optimization
            U_opt, V_opt, W_opt, err = _try_optimize(
                U, V, W, n, m, p, target_rank, deadline, max_time_per_attempt=4.0
            )
            
            if U_opt is not None and err < 1e-8:
                # Verify and return
                valid, _ = _verify(U_opt, V_opt, W_opt, n, m, p)
                if valid:
                    best_rank = target_rank
                    best_result = _to_lists(U_opt, V_opt, W_opt)
                    return best_result
            
            # Try snapping to half-integers if close
            if U_opt is not None and err < 0.5:
                U_snap = np.round(U_opt * 2) / 2.0
                V_snap = np.round(V_opt * 2) / 2.0
                W_snap = np.round(W_opt * 2) / 2.0
                valid, _ = _verify(U_snap, V_snap, W_snap, n, m, p)
                if valid:
                    best_rank = target_rank
                    best_result = _to_lists(U_snap, V_snap, W_snap)
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
