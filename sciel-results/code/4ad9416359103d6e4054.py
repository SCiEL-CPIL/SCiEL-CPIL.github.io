"""Matrix multiplication 2x2x15 tensor decomposition using Strassen blocks with rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """
    Build schoolbook rank-4 decomposition for 2x2 x 2x1 matrix multiplication.
    Returns U (4x4), V (2x4), W (2x4).
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_initial_solution():
    """Build the rank-53 solution using Strassen blocks."""
    n, m, p = 2, 2, 15
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    col_offset = 0
    for b in range(7):
        start_k = b * 2
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for kl in range(2):
                V[j * p + start_k + kl, col_offset:col_offset+7] = V_str[j*2+kl, :]
        for kl in range(2):
            for i in range(n):
                W[(start_k+kl) * n + i, col_offset:col_offset+7] = W_str[kl*2+i, :]
        col_offset += 7
    
    # Last block (k=14)
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]
    
    return U, V, W

def _try_reduce_rank(U, V, W, target_rank, n, m, p, max_iters=300, tol=1e-10):
    """Try to reduce to target_rank using alternating least squares."""
    U_new = U[:, :target_rank].copy()
    V_new = V[:, :target_rank].copy()
    W_new = W[:, :target_rank].copy()
    
    for it in range(max_iters):
        # Update U: for each (i,j), solve M @ x = ones(p)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V_new[j*p+k, :] * W_new[k*n+i, :]
                try:
                    U_new[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve M @ x = ones(n)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U_new[i*m+j, :] * W_new[k*n+i, :]
                try:
                    V_new[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve M @ x = ones(m)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                try:
                    W_new[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        if _verify(U_new, V_new, W_new, n, m, p, tol):
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build initial rank-53 solution
    U, V, W = _build_initial_solution()
    
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    for target_rank in range(52, 44, -1):
        if time.time() - start_time > 100:
            break
        
        result = _try_reduce_rank(U, V, W, target_rank, n, m, p, max_iters=200)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            # Update starting point for next reduction
            U, V, W = result
    
    # Verify final result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to initial solution
        best_U, best_V, best_W = _build_initial_solution()
        best_rank = 53
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
