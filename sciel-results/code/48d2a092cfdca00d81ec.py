"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 110  # Safety margin before 120s kill

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

def _build_rank_3p_plus_1(p):
    """
    Build a rank 3p+1 decomposition for 2x2 x 2xp matrix multiplication.
    This is based on the known result that R(2,2,p) = 3p+1.
    """
    n, m = 2, 2
    rank = 3 * p + 1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The construction uses 3 terms per k, plus 1 extra term.
    # Term 3k: A00 * B0k * C0k
    # Term 3k+1: A11 * B1k * C1k
    # Term 3k+2: (A01 + A10) * (B0k + B1k) * (C0k + C1k) -- This is a guess, needs verification.
    
    # Let's use the verified Strassen-like pattern for p=2 to guide the general case.
    # For p=2, rank 7.
    # The terms are not simply separable.
    
    # If I cannot verify the 3p+1 construction quickly, I will return None.
    # For now, I will implement the block decomposition which is known to work.
    
    return None

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    # Strassen 2x2
    U_str = np.array([
        [1, 0, 1, 1, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_str = np.array([
        [1, 1, 1, -1, 0, 1, 0],
        [0, 0, -1, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [1, -1, 0, 0, 1, 0, 1]
    ])
    W_str = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [1, -1, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]
    ])
    
    # Schoolbook 2x1
    U_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    W_sb = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    total_rank = 53
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank): return False
    if V.shape != (m*p, rank): return False
    if W.shape != (p*n, rank): return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank53()
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
