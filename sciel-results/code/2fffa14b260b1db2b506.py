import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _is_valid_solution(U, V, W, n, m, p, tol=1e-9):
    """
    Checks if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    """
    R = U.shape[1]
    # Convert to numpy arrays for speed
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_solution(n, m, p):
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

def _try_find_lower_rank(n, m, p, target_rank, max_time=10.0, max_iter=500, seed=42):
    """
    Attempts to find a decomposition of rank < n*m*p using ALS.
    Returns None if not found.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Try a few random initializations
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
            
        # Initialize with random values, maybe sparse
        U = np.random.rand(n * m, target_rank) * 0.1
        V = np.random.rand(m * p, target_rank) * 0.1
        W = np.random.rand(p * n, target_rank) * 0.1
        
        # Add some structure: force some entries to be 0 or 1?
        # For now, just random.
        
        converged = False
        for it in range(max_iter):
            if time.time() - start_time > max_time:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Sum over k: sum_k V[j*p+k, :] * W[k*n+i, :]
                    # This vector is (target_rank,)
                    rhs_vec = np.zeros(target_rank)
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        rhs_vec += V[v_idx, :] * W[w_idx, :]
                    # We want U[u_idx, :] such that dot(U[u_idx, :], rhs_vec) = 1?
                    # No, the equation is sum_c U[u_idx, c] * (sum_k V[v_idx, c] * W[w_idx, c]) = 1
                    # This is a single linear equation for the vector U[u_idx, :].
                    # It's underdetermined. We can project to minimize norm or something.
                    # Actually, ALS usually minimizes sum of squared errors for *all* equations.
                    # But here each U row appears in exactly 'p' equations.
                    # Let's minimize the error of the 'p' equations involving U[u_idx, :].
                    # E = sum_k ( U[u_idx, :] dot (V[v_k, :] * W[w_k, :]) - 1 )^2
                    # This is a quadratic form: A U U^T A^T ...
                    # Simpler: Solve least squares for U[u_idx, :] given V, W.
                    # System: M * U_col = 1, where M is p x R.
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        M[k, :] = V[v_idx, :] * W[w_idx, :]
                    try:
                        U[u_idx, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        M[i, :] = U[u_idx, :] * W[w_idx, :]
                    try:
                        V[v_idx, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        M[j, :] = U[u_idx, :] * V[v_idx, :]
                    try:
                        W[w_idx, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass

            # Check convergence / exactness
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        err += (val - 1.0) ** 2
            if err < 1e-12:
                converged = True
                break

        if converged:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _is_valid_solution(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            else:
                # Maybe the raw float is "exact" enough? No, task requires integer/half-integer.
                # If it doesn't snap, this attempt fails.
                pass
                
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 27 # Hypothesized minimal rank
    
    # Try to find a better solution quickly
    # Use a short time budget for search
    start = time.time()
    search_time_budget = 10.0
    
    sol = _try_find_lower_rank(n, m, p, target_rank, max_time=search_time_budget, max_iter=200)
    
    if sol is not None:
        return sol
        
    # Fallback to schoolbook
    return _schoolbook_solution(n, m, p)
# EVOLVE-BLOCK-END
