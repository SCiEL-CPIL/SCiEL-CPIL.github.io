import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_decomposition_error(U, V, W, T):
    """
    Compute the error between the tensor product sum and T.
    """
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    error = np.linalg.norm(reconstructed - T)
    return error

def _check_exact(U, V, W, n, m, p):
    """
    Check if the decomposition is exact for all (i,j,k) positions.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    # Also check a sample of zero positions
    zero_positions = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for u2 in range(n * m):
                    for v2 in range(m * p):
                        for w2 in range(p * n):
                            u_orig = i * m + j
                            v_orig = j * p + k
                            w_orig = k * n + i
                            if u2 != u_orig or v2 != v_orig or w2 != w_orig:
                                zero_positions.append((u2, v2, w2))
    # Check a subset
    if len(zero_positions) > 100:
        sample = random.sample(zero_positions, 100)
    else:
        sample = zero_positions
    for u, v, w in sample:
        val = 0.0
        for r in range(U.shape[1]):
            val += U[u, r] * V[v, r] * W[w, r]
        if abs(val) > 1e-9:
            return False
    return True

def _try_rank_with_search(R, n, m, p, T, time_limit):
    """
    Try to find a decomposition of rank R using structured search.
    """
    start_time = time.time()
    values = [-1, 0, 1]
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 50
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random values
        U = np.random.choice(values, size=(n * m, R))
        V = np.random.choice(values, size=(m * p, R))
        W = np.random.choice(values, size=(p * n, R))
        
        current_error = _compute_decomposition_error(U, V, W, T)
        
        # Local search
        max_iter = 2000
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if current_error == 0.0:
                break
            
            # Pick a random variable to change
            total_vars = U.size + V.size + W.size
            var_idx = random.randint(0, total_vars - 1)
            
            if var_idx < U.size:
                mat = U
                idx = var_idx
            elif var_idx < U.size + V.size:
                mat = V
                idx = var_idx - U.size
            else:
                mat = W
                idx = var_idx - U.size - V.size
                
            old_val = mat.flat[idx]
            new_val = random.choice(values)
            
            mat.flat[idx] = new_val
            new_error = _compute_decomposition_error(U, V, W, T)
            
            if new_error < current_error:
                current_error = new_error
            else:
                mat.flat[idx] = old_val
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error == 0.0:
            break
    
    if best_error < 1e-10:
        return best_U, best_V, best_W
    return None

def _get_schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    for R in range(12, best_rank):
        if time.time() - start_time > deadline:
            break
            
        time_budget = min(10.0, (deadline - (time.time() - start_time)) / max(best_rank - R, 1))
        if time_budget < 1.0:
            break
            
        result = _try_rank_with_search(R, n, m, p, T, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify exactness
            if _check_exact(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
