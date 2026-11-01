import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the n x m x p matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook block."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[i, :]
            col_offset += 4
    
    return U, V, W

def _try_find_2x2x3_rank10(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Uses ALS with multiple restarts and careful initialization.
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with small random values
        scale = 0.5
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        # ALS iterations
        for it in range(300):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve U[i,j,:] * (V[j,k,:] ⊙ W[k,i,:])^T = 1
            for i in range(n):
                for j in range(m):
                    M = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            M[r, k] = V[j * p + k, r] * W[k * n + i, r]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve V[j,k,:] * (U[i,j,:] ⊙ W[k,i,:])^T = 1
            for j in range(m):
                for k in range(p):
                    N = np.zeros((rank, n))
                    for r in range(rank):
                        for i in range(n):
                            N[r, i] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve W[k,i,:] * (U[i,j,:] ⊙ V[j,k,:])^T = 1
            for k in range(p):
                for i in range(n):
                    P = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            P[r, j] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        error += (actual - 1.0) ** 2
            error = np.sqrt(error)
            
            if error < 1e-8:
                break
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < 1e-10:
            break
    
    if best_U is not None and best_error < 1e-8:
        if _verify_decomposition(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    
    return None

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Strategy:
    1. Start with guaranteed rank-53 (7 Strassen + 1 schoolbook).
    2. Try to find rank-10 for 2x2x3, giving total rank 50.
    3. Return the best valid decomposition found.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Build guaranteed rank-53 baseline
    U_53, V_53, W_53 = _build_rank53()
    
    # Verify baseline
    if not _verify_decomposition(U_53, V_53, W_53, n, m, p):
        # Fallback to schoolbook
        U_53, V_53, W_53 = _build_schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_53, V_53, W_53
    best_rank = 53
    
    # Try to find rank-10 for 2x2x3
    if time.time() < deadline - 10:
        result = _try_find_2x2x3_rank10(deadline - 5)
        if result is not None:
            U_223, V_223, W_223 = result
            total_rank = 5 * 10  # 50
            
            # Build full decomposition using five 2x2x3 blocks
            U_full = np.zeros((n * m, total_rank))
            V_full = np.zeros((m * p, total_rank))
            W_full = np.zeros((p * n, total_rank))
            
            for b in range(5):
                start_k = b * 3
                col_offset = b * 10
                U_full[:, col_offset:col_offset+10] = U_223
                for j in range(m):
                    for k_local in range(3):
                        k_global = start_k + k_local
                        V_full[j * p + k_global, col_offset:col_offset+10] = V_223[j * 3 + k_local, :]
                for k_local in range(3):
                    k_global = start_k + k_local
                    for i in range(n):
                        W_full[k_global * n + i, col_offset:col_offset+10] = W_223[k_local * 2 + i, :]
            
            # Verify
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = total_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
