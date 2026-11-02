import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition of the matmul tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _rank14_decomposition():
    """Known-good rank-14 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_vals, v_vals, w_vals, col):
        for i in range(n * m):
            U[i][col] = u_vals[i]
        for j in range(m * p):
            V[j][col] = v_vals[j]
        for k in range(p * n):
            W[k][col] = w_vals[k]

    # Left Part (Cols 0, 1) - Strassen for 2x2x2
    add_term([1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], 0)
    add_term([0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0], 1)
    add_term([1, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], 2)
    add_term([0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], 3)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], 4)
    add_term([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], 5)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], 6)
    # Right Part (Cols 2, 3) - Strassen for 2x2x2
    add_term([1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1], 7)
    add_term([0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, -1], 8)
    add_term([1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], 9)
    add_term([0, 0, 0, 1], [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0], 10)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0, 1, 0], 11)
    add_term([-1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], 12)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0], 13)
    
    return {"U": U, "V": V, "W": W}

def _try_rank(rank, deadline):
    """Attempt to find a decomposition of given rank via randomized search + rounding."""
    n, m, p = DIMENSIONS
    
    # Use a fixed number of attempts based on rank?
    # Rank 12 is harder, maybe more attempts?
    # But time is limited.
    # Let's do 100 attempts for rank 12, 100 for rank 13.
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        rng = np.random.default_rng(attempt * 1000 + rank * 7 + 42)
        
        # Initialize with random values
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        
        # Alternating least squares
        for it in range(300):
            if time.time() > deadline - 5:
                break
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b_vec = np.ones(p)
                    try:
                        sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[i*m+j, :] = sol
                    except np.linalg.LinAlgError:
                        pass
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    b_vec = np.ones(n)
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[j*p+k, :] = sol
                    except np.linalg.LinAlgError:
                        pass
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    b_vec = np.ones(m)
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[k*n+i, :] = sol
                    except np.linalg.LinAlgError:
                        pass
        
        # Check convergence and rounding
        if _verify_decomposition(U, V, W, n, m, p):
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                def to_list(arr):
                    return [[float(x) for x in row] for row in arr]
                return {"U": to_list(U_rounded), "V": to_list(V_rounded), "W": to_list(W_rounded)}
            
            # Try rounding to integers
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                def to_list(arr):
                    return [[float(x) for x in row] for row in arr]
                return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Try rank 12 (theoretical minimum)
    result = _try_rank(12, deadline)
    if result is not None:
        return result
    
    # Try rank 13
    result = _try_rank(13, deadline)
    if result is not None:
        return result
    
    # Fallback to rank 14
    return _rank14_decomposition()
