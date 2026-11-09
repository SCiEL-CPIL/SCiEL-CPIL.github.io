import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W, rank

def _compute_tensor_fast(U, V, W, n, m, p):
    U_reshaped = U.reshape(n, m, -1) 
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)

def _adam_step(params, grads, m_params, v_params, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Apply Adam update to parameters."""
    for i in range(len(params)):
        m_params[i] = beta1 * m_params[i] + (1 - beta1) * grads[i]
        v_params[i] = beta2 * v_params[i] + (1 - beta2) * grads[i] ** 2
        m_hat = m_params[i] / (1 - beta1 ** t)
        v_hat = v_params[i] / (1 - beta2 ** t)
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

def _try_adam_drop(U, V, W, n, m, p, max_iter=50000, tol=1e-5, deadline=None, lr=0.01):
    """
    Adam-based local search to recover the tensor after dropping a column.
    Uses vectorized gradients for speed.
    """
    rank = U.shape[1]
    T_target = np.ones((n, m, p))
    
    # Flatten parameters for Adam
    U_flat = U.copy().flatten()
    V_flat = V.copy().flatten()
    W_flat = W.copy().flatten()
    
    # Adam state
    m_U = np.zeros_like(U_flat)
    v_U = np.zeros_like(U_flat)
    m_V = np.zeros_like(V_flat)
    v_V = np.zeros_like(V_flat)
    m_W = np.zeros_like(W_flat)
    v_W = np.zeros_like(W_flat)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    U_r = U_flat.reshape(n, m, -1)
    V_r = V_flat.reshape(m, p, -1)
    W_r = W_flat.reshape(p, n, -1)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        T_decomp = _compute_tensor_fast(U_r, V_r, W_r, n, m, p)
        error_max = np.max(np.abs(T_decomp - T_target))
        
        if error_max < tol:
            # Round to nearest 0.5 to clean up noise if close to exact
            U_r = np.round(U_r * 2) / 2
            V_r = np.round(V_r * 2) / 2
            W_r = np.round(W_r * 2) / 2
            return U_r.reshape(U.shape), V_r.reshape(V.shape), W_r.reshape(W.shape), True
        
        if error_max < best_error:
            best_error = error_max
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        
        # Vectorized Gradient Computation
        # diff[i, j, k] = T[i, j, k] - 1
        diff = T_decomp - 1.0
        
        # grad_U[i, j, c] = sum_k diff[i, j, k] * V_r[j, k, c] * W_r[k, i, c]
        # We can compute this using einsum
        # V_r[j, k, c] and W_r[k, i, c]
        # Let's try:
        # grad_U[i, j, c] = sum_k diff[i, j, k] * V_r[j, k, c] * W_r[k, i, c]
        # This is a contraction over k.
        # diff: (n, m, p)
        # V_r: (m, p, rank)
        # W_r: (p, n, rank)
        
        # Term1: diff[i, j, k] * V_r[j, k, c] -> (n, m, p, rank) is too big?
        # n=3, m=3, p=5, rank~45. 3*3*5*45 = 2025. Small.
        # Let's just do it.
        
        # Actually, let's use einsum for efficiency and clarity.
        # grad_U[i, j, c] = sum_k diff[i, j, k] V[j, k, c] W[k, i, c]
        # 'ijk,jkc,kic->ijc'
        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        
        # grad_V[j, k, c] = sum_i diff[i, j, k] U[i, j, c] W[k, i, c]
        # 'ijk,ijc,kic->jkc'
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        
        # grad_W[k, i, c] = sum_j diff[i, j, k] U[i, j, c] V[j, k, c]
        # 'ijk,ijc,jkc->kic'
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        # Apply Adam update
        t_adam = it + 1
        _adam_step([U_flat], [grad_U.flatten()], [m_U], [v_U], t_adam, lr)
        _adam_step([V_flat], [grad_V.flatten()], [m_V], [v_V], t_adam, lr)
        _adam_step([W_flat], [grad_W.flatten()], [m_W], [v_W], t_adam, lr)
        
        # Update views
        U_r = U_flat.reshape(n, m, -1)
        V_r = V_flat.reshape(m, p, -1)
        W_r = W_flat.reshape(p, n, -1)
    
    U_final = U_flat.reshape(U.shape)
    V_final = V_flat.reshape(V.shape)
    W_final = W_flat.reshape(W.shape)
    
    T_final = _compute_tensor_fast(U_final, V_final, W_final, n, m, p)
    if np.allclose(T_final, T_target, atol=tol):
        return U_final, V_final, W_final, True
    return best_U, best_V, best_W, False

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_sb
    
    # Attempt to drop columns one by one
    # We try to drop a column from the current best solution
    # If successful, we try to drop another.
    
    # We need to be careful with which column to drop.
    # In schoolbook, all columns are symmetric.
    # In a reduced rank solution, some columns might be more "redundant".
    # However, for a first pass, dropping column 0 is a good heuristic.
    # Or we can try all columns if time permits.
    
    current_U = best_U
    current_V = best_V
    current_W = best_W
    current_rank = best_rank
    
    # Try to drop up to 5 columns (or until time runs out)
    # Since the tensor is small, we can afford to try a few different drop indices
    # if the first one fails.
    
    max_attempts = 5
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
            
        # Try dropping column 0 (or a random one if we were more advanced)
        # For now, let's stick to column 0 as it's deterministic and simple.
        # If it fails, we might try column 1, etc.
        drop_idx = 0
        
        U_test = current_U.copy()
        V_test = current_V.copy()
        W_test = current_W.copy()
        
        # Zero out the column
        U_test[:, drop_idx] = 0.0
        V_test[:, drop_idx] = 0.0
        W_test[:, drop_idx] = 0.0
        
        # Run Adam-based local search to recover
        U_res, V_res, W_res, success = _try_adam_drop(
            U_test, V_test, W_test, n, m, p, 
            max_iter=10000, # Increased due to vectorization
            deadline=deadline - 10,
            lr=0.01
        )
        
        if success:
            current_U, current_V, current_W = U_res, V_res, W_res
            current_rank -= 1
            best_U, best_V, best_W, best_rank = U_res, V_res, W_res, current_rank
            # Continue loop to try dropping another column
        else:
            # If it failed, break out. We don't have a good strategy for trying other columns
            # without significantly increasing complexity and time, and the first drop
            # is usually the hardest or easiest depending on the landscape.
            # Given the time constraint, we assume if col 0 fails, others might too,
            # or the landscape is such that we need a different approach.
            break
            
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
