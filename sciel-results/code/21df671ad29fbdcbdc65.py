"""Matrix multiplication 2x2x15 tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

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

def _build_strassen_2x2x3():
    """
    Build rank-10 decomposition for 2x2 x 2x3 matrix multiplication.
    Uses Strassen for the first 2 columns and schoolbook for the 3rd.
    Returns U (4x10), V (6x10), W (6x10).
    """
    n, m, p = 2, 2, 3
    # Use Strassen(2x2x2) + schoolbook(2x2x1)
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    rank = 7 + 4  # 11... wait, let me think about this differently.
    # Actually for 2x2x3, we can use a direct construction.
    # The standard result is rank(2x2x3) <= 10.
    # Let me construct it: use Strassen for columns 0,1 and schoolbook for column 2.
    # That gives 7+4=11. But there's a rank-10 construction.
    # For now, let's use the 11-rank version and see if we can do better.
    
    rank_total = 7 + 4
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Strassen part for k=0,1
    U[:, 0:7] = U_str
    for j in range(m):
        for k in range(2):
            V[j * p + k, 0:7] = V_str[j * 2 + k, :]
    for k in range(2):
        for i in range(n):
            W[k * n + i, 0:7] = W_str[k * 2 + i, :]
    
    # Schoolbook part for k=2
    U[:, 7:11] = U_sb
    for j in range(m):
        V[j * p + 2, 7:11] = V_sb[j * 1, :]
    for i in range(n):
        W[2 * n + i, 7:11] = W_sb[0 * 2 + i, :]
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    # Check shapes
    assert U.shape == (n*m, rank), f"U shape {U.shape} != {(n*m, rank)}"
    assert V.shape == (m*p, rank), f"V shape {V.shape} != {(m*p, rank)}"
    assert W.shape == (p*n, rank), f"W shape {p*n, rank} != {W.shape}"
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i,j,k valid, else 0
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] should be 1 for all valid i,j,k (since each product term contributes)
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) term has coefficient 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                if abs(val - expected) > 1e-10:
                    return False, (i, j, k, val, expected)
    return True, None

def _build_block_decomposition(n, m, p, block_sizes):
    """
    Build a block diagonal decomposition using Strassen for 2-wide blocks
    and schoolbook for 1-wide blocks.
    Returns (U, V, W, total_rank) or None if invalid.
    """
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    total_rank = 0
    for bs in block_sizes:
        if bs == 2:
            total_rank += 7
        elif bs == 1:
            total_rank += 4
        else:
            return None
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(len(block_sizes)):
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
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W, total_rank

def _try_all_partitions(n, m, p, deadline):
    """
    Try all valid partitions of p into 1s and 2s, compute the rank,
    and return the best (lowest rank) valid decomposition.
    """
    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    # Generate all compositions of p using 1s and 2s
    # This is equivalent to choosing how many 2s to use (0 to p//2)
    # and the rest are 1s. But the order matters for the block structure.
    # Actually, since all blocks of the same size are identical, only the count matters.
    # Let's enumerate by number of 2-blocks.
    
    for num_2s in range(p // 2 + 1):
        num_1s = p - 2 * num_2s
        if num_1s < 0:
            continue
        total_rank = 7 * num_2s + 4 * num_1s
        
        if total_rank >= best_rank:
            continue
        
        # Check time
        if _time_remaining() < 5.0:
            break
        
        # Build block sizes: all 2s first, then all 1s (order doesn't matter for rank)
        block_sizes = [2] * num_2s + [1] * num_1s
        
        result = _build_block_decomposition(n, m, p, block_sizes)
        if result is None:
            continue
        U, V, W, rank = result
        
        # Verify
        valid, info = _verify(U, V, W, n, m, p)
        if valid:
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the known good partition: seven 2s and one 1 (rank 53)
    block_sizes_default = [2, 2, 2, 2, 2, 2, 2, 1]
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    total_rank_default = 7 * 7 + 4  # 53
    U = np.zeros((n * m, total_rank_default))
    V = np.zeros((m * p, total_rank_default))
    W = np.zeros((p * n, total_rank_default))
    
    col_offset = 0
    for b in range(len(block_sizes_default)):
        block_size = block_sizes_default[b]
        start_k = sum(block_sizes_default[:b])
        
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
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    # Verify the default
    valid, info = _verify(U, V, W, n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, total_rank_default
    
    if not valid:
        # Fall back to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = np.zeros((n * m, total_rank_fb))
        V_fb = np.zeros((m * p, total_rank_fb))
        W_fb = np.zeros((p * n, total_rank_fb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, idx] = 1
                    V_fb[j*p+k, idx] = 1
                    W_fb[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Now try other partitions to see if we can do better
    if _time_remaining() > 10.0:
        best_U2, best_V2, best_W2, best_rank2 = _try_all_partitions(n, m, p, _TIME_LIMIT)
        if best_U2 is not None and best_rank2 < best_rank:
            best_U, best_V, best_W, best_rank = best_U2, best_V2, best_W2, best_rank2
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
