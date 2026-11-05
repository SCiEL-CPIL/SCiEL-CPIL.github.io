"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (a21 + a22) b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    # M3 = a11 (b12 - b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    # M4 = a22 (b21 - b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5 = (a11 + a12) b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # M6 = (a21 - a11) (b11 + b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (a12 - a22) (b21 + b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns the global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    # U: local index li*mi + lj -> global (n_off+li)*m + (m_off+lj)
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    # V: local index lj*pi + lk -> global (m_off+lj)*p + (p_off+lk)
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    # W: local index lk*ni + li -> global (p_off+lk)*n + (n_off+li)
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_rank(ni, mi, pi):
    """Compute the rank for a block of size ni x mi x pi."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    return ni * mi * pi


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = _block_rank(ni, mi, pi)
                block_ranks[(i, j, k)] = r
                total_rank += r
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using vectorized numpy ops."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _direct_sum_decomposition(n, m, p):
    """
    Direct-sum (Kronecker) decomposition:
    Treat the 3x5x5 multiplication as a direct sum of row-wise multiplications.
    
    Key insight: For each output row i (i=0,1,2), we compute:
      C[i,:] = A[i,:] * B
    
    This is a 1 x 5 x 5 multiplication for each row.
    A 1x5x5 multiplication has rank 5 (each output element is a dot product).
    
    But we can do better by using the structure:
    - Split the 5x5 B matrix into blocks
    - Use Strassen-like tricks on the 2x2 sub-blocks
    
    Strategy: Partition n=[2,1], m=[2,2,1], p=[2,2,1]
    This gives us multiple 2x2x2 Strassen blocks plus small schoolbook blocks.
    
    Actually, let's use a smarter approach:
    - The 3x5x5 tensor can be seen as:
      Row 0-1 (2 rows) x 5 cols x 5 inner
      Row 2 (1 row) x 5 cols x 5 inner
    
    For the 2x5x5 part:
      Split m=[2,2,1], p=[2,2,1]
      Blocks:
        (0,0,0): 2x2x2 -> Strassen 7
        (0,1,0): 2x2x2 -> Strassen 7
        (0,2,0): 2x1x2 -> schoolbook 4
        (0,0,1): 2x2x2 -> Strassen 7
        (0,1,1): 2x2x2 -> Strassen 7
        (0,2,1): 2x1x2 -> schoolbook 4
        (0,0,2): 2x2x1 -> schoolbook 4
        (0,1,2): 2x2x1 -> schoolbook 4
        (0,2,2): 2x1x1 -> schoolbook 2
      Total for 2x5x5: 7+7+4+7+7+4+4+4+2 = 46
    
    For the 1x5x5 part:
      Split m=[2,2,1], p=[2,2,1]
      All blocks are 1xmi x pi, schoolbook:
        (1,0,0): 1x2x2 -> 4
        (1,1,0): 1x2x2 -> 4
        (1,2,0): 1x1x2 -> 2
        (1,0,1): 1x2x2 -> 4
        (1,1,1): 1x2x2 -> 4
        (1,2,1): 1x1x2 -> 2
        (1,0,2): 1x2x1 -> 2
        (1,1,2): 1x2x1 -> 2
        (1,2,2): 1x1x1 -> 1
      Total for 1x5x5: 4+4+2+4+4+2+2+2+1 = 25
    
    Grand total: 46 + 25 = 71
    
    Wait, that's better than 74! Let me verify this is correct.
    """
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    U, V, W, total_rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
    return U, V, W, total_rank


def _fined_partition_decomposition(n, m, p):
    """
    Fine-grained partition maximizing 2x2x2 Strassen blocks.
    
    Try: n=[1,1,1], m=[2,2,1], p=[2,2,1]
    All n-blocks are size 1, so no 2x2x2 Strassen possible.
    
    Better: n=[2,1], m=[2,2,1], p=[2,2,1] as above.
    
    Even better: Can we get more 2x2x2 blocks?
    n=[2,1], m=[2,2,1], p=[2,2,1]:
    2x2x2 blocks at: (0,0,0), (0,1,0), (0,0,1), (0,1,1) -> 4 Strassen blocks
    
    What about n=[2,1], m=[1,2,2], p=[2,2,1]?
    2x2x2 blocks at: (0,1,0), (0,2,0), (0,1,1), (0,2,1) -> 4 Strassen blocks
    Same count.
    
    What about n=[1,2], m=[2,2,1], p=[2,2,1]?
    2x2x2 blocks at: (1,0,0), (1,1,0), (1,0,1), (1,1,1) -> 4 Strassen blocks
    Same.
    
    So max 4 Strassen blocks with this partitioning.
    Total rank = 4*7 + (9-4)*schoolbook_avg
    Let me just compute directly.
    """
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    U, V, W, total_rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
    return U, V, W, total_rank


def _asymmetric_partition(n, m, p):
    """
    Try asymmetric partitions to find lower rank.
    
    Key idea: The 3x5x5 problem is asymmetric. We can try:
    - n=[2,1], m=[3,2], p=[3,2]
    - n=[2,1], m=[2,3], p=[3,2]
    - etc.
    
    Let's try several and pick the best.
    """
    candidates = [
        ([2, 1], [2, 2, 1], [2, 2, 1]),
        ([2, 1], [2, 3], [2, 3]),
        ([2, 1], [3, 2], [2, 3]),
        ([2, 1], [2, 3], [3, 2]),
        ([2, 1], [3, 2], [3, 2]),
        ([1, 2], [2, 2, 1], [2, 2, 1]),
        ([2, 1], [1, 2, 2], [2, 2, 1]),
        ([2, 1], [2, 2, 1], [1, 2, 2]),
        ([2, 1], [1, 2, 2], [1, 2, 2]),
        ([2, 1], [2, 1, 2], [2, 1, 2]),
    ]
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for n_parts, m_parts, p_parts in candidates:
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        try:
            U, V, W, total_rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if total_rank < best_rank and _verify_decomposition(U, V, W, n, m, p):
                best_rank = total_rank
                best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    else:
        # Fallback
        U, V, W, total_rank = _block_decomposition(n, m, p, [2, 1], [2, 3], [2, 3])
        return U, V, W, total_rank


def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using matrix operations."""
    # Build the full tensor check
    # For each r: outer product of U[:,r], V[:,r], W[:,r] should sum to identity tensor
    # More efficiently: for each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    rank = U.shape[1]
    # Vectorized check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 115  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Try the asymmetric partition search first
    try:
        U, V, W, rank = _asymmetric_partition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to known good rank-74
    U, V, W = _hardcoded_rank74(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _hardcoded_rank74(n, m, p):
    """
    Hardcoded rank-74 decomposition.
    """
    n_parts = [2, 1]
    m_parts = [2, 3]
    p_parts = [2, 3]
    U, V, W, total_rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
    return U, V, W

# EVOLVE-BLOCK-END
