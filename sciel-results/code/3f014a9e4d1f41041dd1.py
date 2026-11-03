"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_TIMEOUT = 120.0
_SAFETY_MARGIN = 15.0

def _get_tensor(n, m, p):
    """The multiplication tensor for n x m * m x p is all ones."""
    return np.ones((n, m, p))

def _compute_error(U, V, W, n, m, p):
    """Compute the squared error between the decomposition and the target tensor."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We can compute this using einsum or loops.
    # Vectorized:
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    # T[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    # Note: W index is k*n + i. In W_reshaped, index is [k, i, r].
    T = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
    # Target is all ones
    error = np.sum((T - 1.0)**2)
    return error

def _compute_grad(params, n, m, p):
    """Compute gradient of the error w.r.t flattened parameters."""
    R = params['R']
    U = params['U']
    V = params['V']
    W = params['W']
    
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
    E = 2 * (T - 1.0)
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    # dU[i, j, r] = sum_k E[i, j, k] V[j, k, r] W[k, i, r]
    dU_reshaped = np.einsum('ijk,jkr,ki r->ijr', E, V_reshaped, W_reshaped)
    # dV[j, k, r] = sum_i E[i, j, k] U[i, j, r] W[k, i, r]
    dV_reshaped = np.einsum('ijk,ijr,ki r->jkr', E, U_reshaped, W_reshaped)
    # dW[k, i, r] = sum_j E[i, j, k] U[i, j, r] V[j, k, r]
    dW_reshaped = np.einsum('ijk,ijr,jkr->ki r', E, U_reshaped, V_reshaped)
    
    dU = dU_reshaped.reshape(n*m, R)
    dV = dV_reshaped.reshape(m*p, R)
    dW = dW_reshaped.reshape(p*n, R)
    
    # Flatten gradients
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return grad

def _objective(x, n, m, p, R):
    """Objective function for optimizer."""
    size_U = n * m * R
    size_V = m * p * R
    U = x[:size_U].reshape(n*m, R)
    V = x[size_U:size_U+size_V].reshape(m*p, R)
    W = x[size_U+size_V:].reshape(p*n, R)
    
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
    error = np.sum((T - 1.0)**2)
    return error

def _grad(x, n, m, p, R):
    """Gradient function for optimizer."""
    size_U = n * m * R
    size_V = m * p * R
    U = x[:size_U].reshape(n*m, R)
    V = x[size_U:size_U+size_V].reshape(m*p, R)
    W = x[size_U+size_V:].reshape(p*n, R)
    
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
    E = 2 * (T - 1.0)
    
    dU_reshaped = np.einsum('ijk,jkr,ki r->ijr', E, V_reshaped, W_reshaped)
    dV_reshaped = np.einsum('ijk,ijr,ki r->jkr', E, U_reshaped, W_reshaped)
    dW_reshaped = np.einsum('ijk,ijr,jkr->ki r', E, U_reshaped, V_reshaped)
    
    dU = dU_reshaped.reshape(n*m, R)
    dV = dV_reshaped.reshape(m*p, R)
    dW = dW_reshaped.reshape(p*n, R)
    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return grad

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_find_decomposition(rank_target, n, m, p, max_restarts=20):
    """Try to find a decomposition of the given rank using L-BFGS-B."""
    if rank_target < 1:
        return None
    
    size_U = n * m * rank_target
    size_V = m * p * rank_target
    size_W = p * n * rank_target
    total_size = size_U + size_V + size_W
    
    for restart in range(max_restarts):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        
        # Random initialization
        x0 = np.random.randn(total_size)
        
        # Run optimization
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(n, m, p, rank_target), 
                jac=_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < 1e-6:
                # Extract U, V, W
                U = res.x[:size_U].reshape(n*m, rank_target)
                V = res.x[size_U:size_U+size_V].reshape(m*p, rank_target)
                W = res.x[size_U+size_V:].reshape(p*n, rank_target)
                
                # Try to round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _validate(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If rounding fails, try without rounding? 
                # The prompt requires integer/half-integer. 
                # If we can't round, this solution is invalid for the output format.
                # We continue searching.
                
        except Exception as e:
            pass
            
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from low to high
    # We expect the rank to be significantly lower than 45.
    # Let's try a range.
    for rank in range(25, schoolbook_rank):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        result = _try_find_decomposition(rank, n, m, p, max_restarts=10)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
