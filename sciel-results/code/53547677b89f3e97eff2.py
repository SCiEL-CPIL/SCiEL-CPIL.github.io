import time
import numpy as np

_DIMENSIONS = [4, 4, 8]
_TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p):
    R = U.shape[1]
    tol = 1e-9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _greedy_remove_columns(U, V, W, n, m, p, time_limit):
    """
    Greedily remove columns that are linearly dependent across all three matrices.
    Try all columns, not just the last one.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()
    
    changed = True
    while changed:
        changed = False
        current_rank = U.shape[1]
        if current_rank <= 1:
            break
        if time.time() - start > time_limit:
            break
            
        # Precompute ranks of full matrices
        r_U = np.linalg.matrix_rank(U)
        r_V = np.linalg.matrix_rank(V)
        r_W = np.linalg.matrix_rank(W)
        
        for col in range(current_rank):
            if time.time() - start > time_limit:
                break
            # Check if column `col` is in the span of the other columns
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            
            U_rest = U[:, mask]
            V_rest = V[:, mask]
            W_rest = W[:, mask]
            
            # Check rank
            r_U_rest = np.linalg.matrix_rank(U_rest)
            if r_U_rest < r_U:
                continue
            r_V_rest = np.linalg.matrix_rank(V_rest)
            if r_V_rest < r_V:
                continue
            r_W_rest = np.linalg.matrix_rank(W_rest)
            if r_W_rest < r_W:
                continue
            
            # All three have the same rank with or without this column
            # Remove it
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            changed = True
            # Reset and restart the loop since indices changed
            break
    
    return U, V, W

def _strassen_222():
    """Strassen's algorithm for 2x2x2 matrix multiplication, rank 7."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """Kronecker product decomposition."""
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_decomposition(U1, V1, W1):
    """Direct sum: doubles the problem size in m and p."""
    n, r = U1.shape
    m, _ = V1.shape
    p, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)
    W = np.zeros((2 * p, R), dtype=float)
    
    U[:, :r] = U1
    V[:m, :r] = V1
    W[:p, :r] = W1
    
    U[:, r:] = U1
    V[m:, r:] = V1
    W[p:, r:] = W1
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    R = n * m * p
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj, col] = 1
                V[jj * p + kk, col] = 1
                W[kk * n + ii, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = _DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Strategy 1: Strassen-based structured decomposition (rank 98)
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_decomposition(U444, V444, W444)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try greedy column removal from Strassen-based decomposition
    remaining_time = deadline - time.time() - 5.0  # 5s safety margin
    if remaining_time > 10:
        U_try, V_try, W_try = _greedy_remove_columns(U, V, W, n, m, p, remaining_time)
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            if U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_U, best_V, best_W = U_try, V_try, W_try
    
    # Strategy 2: Try schoolbook + greedy removal (different starting point)
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 10:
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        U_try, V_try, W_try = _greedy_remove_columns(U_s, V_s, W_s, n, m, p, remaining_time)
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            if U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_U, best_V, best_W = U_try, V_try, W_try
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
