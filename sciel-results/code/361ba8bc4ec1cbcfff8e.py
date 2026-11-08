"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _snap_to_half_int(arr):
    """Snap numpy array values to the nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _verify_decomposition_numpy(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using numpy for speed."""
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    return np.allclose(T, 1.0, atol=tol)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly satisfies the tensor equation."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    return _verify_decomposition_numpy(U_arr, V_arr, W_arr, n, m, p, tol)

def _compute_error(U, V, W, n, m, p):
    """Compute max error of the decomposition."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _als_step(U, V, W, n, m, p, rank):
    """Perform one ALS iteration. Returns updated U, V, W."""
    # Update U
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, rank))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                X = np.ones(p) @ np.linalg.pinv(M.T)
                U_new[i*m+j, :] = X
            except:
                pass
    U = U_new
    
    # Update V
    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, rank))
            for i in range(n):
                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                X = np.ones(n) @ np.linalg.pinv(M.T)
                V_new[j*p+k, :] = X
            except:
                pass
    V = V_new
    
    # Update W
    W_new = W.copy()
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                X = np.ones(m) @ np.linalg.pinv(M.T)
                W_new[k*n+i, :] = X
            except:
                pass
    W = W_new
    
    return U, V, W

def _try_rank_als(rank, n, m, p, max_iters=80, n_restarts=15, timeout=None, start_time=None):
    """Try to find a rank-r decomposition using ALS with various initializations."""
    if start_time is None:
        start_time = time.time()
    
    for restart in range(n_restarts):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Small random
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
        elif restart % 3 == 1:
            # Larger random
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
        else:
            # Binary-like initialization
            U = np.random.choice([-1, 1], size=(n * m, rank)) * 0.3
            V = np.random.choice([-1, 1], size=(m * p, rank)) * 0.3
            W = np.random.choice([-1, 1], size=(p * n, rank)) * 0.3
        
        for it in range(max_iters):
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            U, V, W = _als_step(U, V, W, n, m, p, rank)
            
            # Check convergence every few iterations
            if it % 5 == 4:
                max_err = _compute_error(U, V, W, n, m, p)
                if max_err < 0.1:
                    # Try snapping
                    U_snapped = _snap_to_half_int(U)
                    V_snapped = _snap_to_half_int(V)
                    W_snapped = _snap_to_half_int(W)
                    
                    if _verify_decomposition_numpy(U_snapped, V_snapped, W_snapped, n, m, p, 1e-9):
                        return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                
                if max_err < 1e-6:
                    # Very close, try more precise snapping
                    U_snapped = _snap_to_half_int(U)
                    V_snapped = _snap_to_half_int(V)
                    W_snapped = _snap_to_half_int(W)
                    if _verify_decomposition_numpy(U_snapped, V_snapped, W_snapped, n, m, p, 1e-9):
                        return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
    
    return None

def _try_rank_greedy_reduce(n, m, p, start_rank, target_rank, timeout=None, start_time=None):
    """Try to reduce rank greedily by removing columns that have small contribution."""
    if start_time is None:
        start_time = time.time()
    
    # Start with schoolbook
    rank = start_rank
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
    
    # Try to merge/remove columns
    while rank > target_rank:
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Try removing each column and check if we can fix the rest
        found_removal = False
        for col_idx in range(rank - 1, -1, -1):
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            # Remove this column
            U_test = np.delete(U, col_idx, axis=1)
            V_test = np.delete(V, col_idx, axis=1)
            W_test = np.delete(W, col_idx, axis=1)
            
            # Try to fix with ALS
            U_fix = U_test.copy()
            V_fix = V_test.copy()
            W_fix = W_test.copy()
            
            for it in range(20):
                if timeout is not None and time.time() - start_time > timeout:
                    break
                U_fix, V_fix, W_fix = _als_step(U_fix, V_fix, W_fix, n, m, p, rank - 1)
            
            max_err = _compute_error(U_fix, V_fix, W_fix, n, m, p)
            if max_err < 0.5:
                U_snapped = _snap_to_half_int(U_fix)
                V_snapped = _snap_to_half_int(V_fix)
                W_snapped = _snap_to_half_int(W_fix)
                if _verify_decomposition_numpy(U_snapped, V_snapped, W_snapped, n, m, p, 1e-9):
                    U = U_snapped
                    V = V_snapped
                    W = W_snapped
                    rank -= 1
                    found_removal = True
                    break
        
        if not found_removal:
            break
    
    if rank == target_rank:
        U_snapped = _snap_to_half_int(U)
        V_snapped = _snap_to_half_int(V)
        W_snapped = _snap_to_half_int(W)
        if _verify_decomposition_numpy(U_snapped, V_snapped, W_snapped, n, m, p, 1e-9):
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Lower bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    lower_bound = max(n*m, m*p, p*n)
    
    # Strategy 1: Try ALS from lower bound up
    for rank in range(lower_bound, min(lower_bound + 10, rank_school + 1)):
        if time.time() - start_time > deadline:
            break
        
        result = _try_rank_als(rank, n, m, p, max_iters=60, n_restarts=10, 
                              timeout=deadline, start_time=start_time)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # Strategy 2: Try greedy reduction from schoolbook
    if best_rank == rank_school and time.time() - start_time < deadline:
        for target in range(lower_bound, min(lower_bound + 15, rank_school)):
            if time.time() - start_time > deadline:
                break
            result = _try_rank_greedy_reduce(n, m, p, rank_school, target, 
                                             timeout=deadline, start_time=start_time)
            if result is not None:
                best_result = result
                best_rank = target
                break
    
    return best_result
# EVOLVE-BLOCK-END
