import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 9]
DEADLINE = 130  # Safety margin before 150s kill

def _build_multiplication_tensor(n, m, p):
    """Build the multiplication tensor T[i,j,k] = 1 if (i,j) in U, (j,k) in V, (k,i) in W align."""
    # The tensor T has indices (i, j, k) where:
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i,j,k] = 1 means the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # For exact matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(2 * x) / 2.0

def _try_rank_r(U_init, V_init, W_init, n, m, p, r, max_iter=100, tol=1e-8):
    """
    Try to find a rank-r decomposition starting from initial matrices.
    Uses alternating least squares with half-integer rounding.
    """
    # Work with numpy arrays for speed
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    # Ensure shapes
    U = U[:n*m, :r]
    V = V[:m*p, :r]
    W = W[:p*n, :r]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for iteration in range(max_iter):
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                # For each k, we have: sum_r U[u_idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is: sum_r U[u_idx, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # Let M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then we need M @ U[u_idx, :]^T = ones(p)
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve M @ u = ones(p) in least squares sense
                if r > 0:
                    u_new, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[u_idx, :] = u_new
                else:
                    U[u_idx, :] = 0
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                # For each i, we have: sum_r U[i*m+j, r] * V[v_idx, r] * W[k*n+i, r] = 1
                # Let M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then we need M @ V[v_idx, :]^T = ones(n)
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                if r > 0:
                    v_new, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[v_idx, :] = v_new
                else:
                    V[v_idx, :] = 0
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                # For each j, we have: sum_r U[i*m+j, r] * V[j*p+k, r] * W[w_idx, r] = 1
                # Let M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then we need M @ W[w_idx, :]^T = ones(m)
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                if r > 0:
                    w_new, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[w_idx, :] = w_new
                else:
                    W[w_idx, :] = 0
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Round to half-integers and check
    U_rounded = np.round(2 * best_U) / 2.0
    V_rounded = np.round(2 * best_V) / 2.0
    W_rounded = np.round(2 * best_W) / 2.0
    
    # Verify rounded solution
    err_rounded = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                err_rounded += (s - 1.0) ** 2
    err_rounded = np.sqrt(err_rounded)
    
    if err_rounded < 1e-6:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U_rounded]
        V_list = [[float(x) for x in row] for row in V_rounded]
        W_list = [[float(x) for x in row] for row in W_rounded]
        return U_list, V_list, W_list
    else:
        return None

def _make_schoolbook_decomposition(n, m, p, rank):
    """Make a schoolbook-style decomposition with given rank (may be partial)."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use the first 'rank' terms of the full decomposition
    for r in range(min(rank, n * m * p)):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Full schoolbook rank
    full_rank = n * m * p  # 54
    
    # Try to find lower-rank decompositions
    best_result = None
    best_rank = full_rank
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _make_schoolbook_decomposition(n, m, p, full_rank)
    best_result = (U_school.tolist(), V_school.tolist(), W_school.tolist())
    
    # Try ranks from full_rank-1 down to some minimum
    # Limit search based on time
    for r in range(full_rank - 1, max(20, full_rank - 20), -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try multiple initializations
        found = False
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Different initializations
            if trial == 0:
                # Partial schoolbook
                U_init, V_init, W_init = _make_schoolbook_decomposition(n, m, p, r)
            elif trial == 1:
                # Random initialization
                np.random.seed(42 + trial)
                U_init = np.random.randn(n * m, r) * 0.5
                V_init = np.random.randn(m * p, r) * 0.5
                W_init = np.random.randn(p * n, r) * 0.5
            else:
                # Sparse random
                np.random.seed(100 + trial)
                U_init = np.random.rand(n * m, r)
                V_init = np.random.rand(m * p, r)
                W_init = np.random.rand(p * n, r)
                # Make sparse
                U_init[np.random.rand(*U_init.shape) < 0.7] = 0
                V_init[np.random.rand(*V_init.shape) < 0.7] = 0
                W_init[np.random.rand(*W_init.shape) < 0.7] = 0
            
            result = _try_rank_r(U_init, V_init, W_init, n, m, p, r, max_iter=50)
            if result is not None:
                U_found, V_found, W_found = result
                # Verify
                if _verify_decomposition(U_found, V_found, W_found, n, m, p):
                    best_result = (U_found, V_found, W_found)
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at rank r, continue to try lower
            pass
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    else:
        # Fallback to schoolbook
        U_school, V_school, W_school = _make_schoolbook_decomposition(n, m, p, full_rank)
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
