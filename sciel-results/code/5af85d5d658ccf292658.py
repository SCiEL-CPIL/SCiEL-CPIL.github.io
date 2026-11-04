import time
import numpy as np

DIMENSIONS = [3, 4, 6]
TIME_LIMIT = 115.0  # seconds, leave margin for the evaluator's 120s kill

def _verify_fast(U_np, V_np, W_np, n, m, p):
    """Fast verification assuming numpy arrays."""
    rank = U_np.shape[1]
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                error = abs(val - 1.0)
                if error > max_error:
                    max_error = error
                if max_error > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_lower_rank(n, m, p, target_rank, time_budget):
    """
    Attempt to find a decomposition of rank target_rank using iterative
    refinement with random restarts. Returns (U, V, W) or None.
    """
    if time_budget <= 0:
        return None
    
    start_time = time.time()
    n_terms = n * m * p  # 72 terms in the tensor
    
    # Precompute target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    
    for attempt in range(50):
        if time.time() - start_time > time_budget * 0.95:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Iterative optimization
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > time_budget * 0.95:
                break
            
            # Compute gradient: minimize sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # For each term, the contribution to the objective is (s - 1)^2 where s = sum_r U_a[r]*V_b[r]*W_c[r]
            # d/dU_a[r] = 2*(s-1)*V_b[r]*W_c[r]
            
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for (u_idx, v_idx, w_idx) in target_indices:
                s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                factor = 2.0 * (s - 1.0)
                U_grad[u_idx, :] += factor * (V[v_idx, :] * W[w_idx, :])
                V_grad[v_idx, :] += factor * (U[u_idx, :] * W[w_idx, :])
                W_grad[w_idx, :] += factor * (U[u_idx, :] * V[v_idx, :])
            
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Check convergence
            if step % 100 == 0:
                max_err = 0.0
                for (u_idx, v_idx, w_idx) in target_indices:
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
                if max_err < 1e-7:
                    # Round to nearest 0.5 and verify
                    U_r = np.round(U * 2.0) / 2.0
                    V_r = np.round(V * 2.0) / 2.0
                    W_r = np.round(W * 2.0) / 2.0
                    if _verify_fast(U_r, V_r, W_r, n, m, p):
                        return U_r, V_r, W_r
                # Adaptive learning rate
                if max_err > 0.1:
                    lr *= 0.5
                elif max_err < 0.01:
                    lr = max(lr * 0.9, 0.001)
        
        # Final check after each restart
        max_err = 0.0
        for (u_idx, v_idx, w_idx) in target_indices:
            s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            err = abs(s - 1.0)
            if err > max_err:
                max_err = err
        if max_err < 1e-7:
            U_r = np.round(U * 2.0) / 2.0
            V_r = np.round(V * 2.0) / 2.0
            W_r = np.round(W * 2.0) / 2.0
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
    
    return None

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a lower rank decomposition
    # Known lower bounds suggest rank >= 66 for 3x4x6, but let's try a few values
    for target_rank in range(66, schoolbook_rank):
        remaining = TIME_LIMIT - (time.time() - start_time)
        if remaining < 10.0:
            break
        result = _try_lower_rank(n, m, p, target_rank, remaining)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    if not _verify_fast(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook decomposition verification failed")
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
