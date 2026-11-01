import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s
    
    n, m, p = 2, 2, 8
    
    # First, get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()
    
    # Try to find a lower-rank decomposition via structured search
    for target_rank in [27, 26, 25, 24]:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        result = _search_low_rank(target_rank, remaining - 5)
        if result is not None:
            return result
    
    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, nonzero_indices, num_a, num_b, num_c, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (num_a, num_b, rank)
    T_hat = np.einsum('abr,cr->abc', UV, W)  # (num_a, num_b, num_c)
    
    # Check non-zero entries
    for a, b, c in nonzero_indices:
        if abs(T_hat[a, b, c] - 1) > tol:
            return False
    
    # Check zero entries
    nonzero_set = set(nonzero_indices)
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                if (a, b, c) not in nonzero_set:
                    if abs(T_hat[a, b, c]) > tol:
                        return False
    
    return True

def _try_rank_reduction(U, V, W, target_rank, time_budget, start_time):
    """Try to reduce the rank of a valid decomposition by merging/removing terms."""
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = n*m, m*p, p*n
    
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))
    
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try removing columns that contribute least
    # For each column r, compute how much it contributes to the residual
    # if we zero it out
    
    best_U, best_V, best_W = None, None, None
    
    for restart in range(50):
        if time.time() - start_time > time_budget:
            break
        
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Try to merge pairs of columns
        for r1 in range(current_rank - 1):
            if time.time() - start_time > time_budget:
                break
            for r2 in range(r1 + 1, current_rank):
                # Try to see if columns r1 and r2 can be merged
                # by finding a linear combination that preserves the tensor
                # This is complex; instead, try zeroing out one column
                # and adjusting others
                
                # Simple approach: try removing column r2 and see if 
                # we can compensate by adjusting other columns
                U_test = U_try.copy()
                V_test = V_try.copy()
                W_test = W_try.copy()
                
                # Zero out column r2
                U_test[:, r2] = 0
                V_test[:, r2] = 0
                W_test[:, r2] = 0
                
                # Check if still valid
                if _verify_decomposition(U_test, V_test, W_test, nonzero_indices, num_a, num_b, num_c, tol=1e-4):
                    # Valid! Now try to further reduce by checking other columns
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    break
            
            if best_U is not None:
                break
        
        if best_U is not None and best_U.shape[1] - 1 <= target_rank:
            # Trim to target_rank
            U_out = best_U[:, :target_rank]
            V_out = best_V[:, :target_rank]
            W_out = best_W[:, :target_rank]
            
            if _verify_decomposition(U_out, V_out, W_out, nonzero_indices, num_a, num_b, num_c, tol=1e-6):
                # Round to half-integers
                U_out = np.round(U_out * 2) / 2
                V_out = np.round(V_out * 2) / 2
                W_out = np.round(W_out * 2) / 2
                return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    return None

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using structured approach."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the list of (a,b,c) indices where T=1
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j  # A index: row i, col j
                b = j * 8 + k  # B index: row j, col k
                c = k * 2 + i  # C index: col k, row i (convention W[k*n+i])
                nonzero_indices.append((a, b, c))
    
    # Start from the rank-28 Strassen solution and try to reduce
    fallback = _strassen_28()
    U = np.array(fallback["U"])
    V = np.array(fallback["V"])
    W = np.array(fallback["W"])
    
    # Try rank reduction from the Strassen solution
    result = _try_rank_reduction(U, V, W, target_rank, time_budget, start)
    if result is not None:
        return result
    
    # Also try random search with better initialization
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.default_rng(seed=restart)
        
        # Initialize with small integers
        U = rng.integers(-1, 2, size=(num_a, target_rank)).astype(float)
        V = rng.integers(-1, 2, size=(num_b, target_rank)).astype(float)
        W = rng.integers(-1, 2, size=(num_c, target_rank)).astype(float)
        
        # Quick check
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        T_hat = np.einsum('abr,cr->abc', UV, W)
        
        # Check non-zero entries first (faster)
        max_err = 0
        for a, b, c in nonzero_indices:
            err = abs(T_hat[a, b, c] - 1)
            if err > max_err:
                max_err = err
                if max_err > 0.1:
                    break
        
        if max_err < 0.1:
            # More careful check
            if _verify_decomposition(U, V, W, nonzero_indices, num_a, num_b, num_c, tol=1e-6):
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
