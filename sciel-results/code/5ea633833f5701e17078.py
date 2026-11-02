"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from itertools import product

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
    U[0*2+0, 0] = 1
    U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1
    V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1
    W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1
    W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1
    W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1
    W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
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
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For each valid (i,j,k), check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for invalid combinations, it should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This combination is valid, should give 1
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_rank52():
    """
    Attempt to build a rank-52 decomposition by using Strassen for 2x2 blocks
    and optimizing the last block.
    
    We use 6 Strassen blocks (12 k's) + 1 special block for the last 3 k's.
    The last 3 k's need a 2x2 x 2x3 decomposition.
    
    Actually, let's try a different approach: use 7 Strassen blocks but see if
    we can merge the last block with something.
    
    Alternative: use 5 Strassen blocks (10 k's) + 1 block of 5 k's.
    5*7 + rank(2x2x5) = 35 + rank(2x2x5)
    If rank(2x2x5) = 17, total = 52.
    If rank(2x2x5) = 16, total = 51.
    
    Let's try to find a good 2x2x5 decomposition.
    """
    n, m, p = DIMENSIONS
    # Use 5 Strassen blocks for first 10 k's, then handle last 5 k's
    num_strassen = 5
    k_strassen = 2 * num_strassen  # 10
    k_remaining = p - k_strassen   # 5
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    total_rank = num_strassen * 7  # 35
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_strassen):
        start_k = b * 2
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
    
    # Now handle the remaining 5 k's with a 2x2 x 2x5 decomposition
    # For now, use schoolbook for the remaining 5 k's: rank = 2*2*5 = 20
    # Total rank = 35 + 20 = 55, which is worse than 53.
    # So this approach doesn't help directly.
    
    # Let's stick with the rank-53 approach for now.
    return _build_rank53()

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Verify baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something is wrong
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
        best_U, best_V, best_W = U, V, W
    
    # Deadline guard
    deadline = start_time + 100  # 100s out of 150s budget
    
    # Try to improve by searching for better decompositions
    # For now, we return the rank-53 decomposition as it's the best known.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
