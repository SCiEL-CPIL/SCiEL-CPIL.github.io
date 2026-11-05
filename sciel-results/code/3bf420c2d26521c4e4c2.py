"""Combined program: block-partitioned Strassen 2x2x2 with verification and safety margin."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for rank-7 Strassen 2x2x2 decomposition."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
    """Verify that the tensor decomposition is correct."""
    U = np.array(U_mat, dtype=np.int64)
    V = np.array(V_mat, dtype=np.int64)
    W = np.array(W_mat, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]))
                if val != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _embed_strassen_block(U, V, W, col, i_list, j_list, k_list, n, m, p):
    """Embed a Strassen 2x2x2 block into the global factors at column offset col."""
    Us, Vs, Ws = _strassen_2x2x2()
    i0, i1 = i_list
    j0, j1 = j_list
    k0, k1 = k_list
    u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
    v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
    w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]
    for c in range(7):
        for s in range(4):
            U[u_rows[s]][col + c] = Us[s][c]
            V[v_rows[s]][col + c] = Vs[s][c]
            W[w_rows[s]][col + c] = Ws[s][c]
    return col + 7

def _try_partition(n, m, p, n_parts, m_parts, p_parts):
    """
    Try a given partition. Returns (U, V, W, rank) or None if invalid.
    Uses Strassen for 2x2x2 blocks, schoolbook otherwise.
    """
    # Compute offsets
    n_off = [0]
    for i in range(len(n_parts) - 1):
        n_off.append(n_off[-1] + n_parts[i])
    m_off = [0]
    for i in range(len(m_parts) - 1):
        m_off.append(m_off[-1] + m_parts[i])
    p_off = [0]
    for i in range(len(p_parts) - 1):
        p_off.append(p_off[-1] + p_parts[i])
    
    # Compute total rank
    rank = 0
    for bi in range(len(n_parts)):
        for bj in range(len(m_parts)):
            for bk in range(len(p_parts)):
                ns = n_parts[bi]
                ms = m_parts[bj]
                ps = p_parts[bk]
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                else:
                    rank += ns * ms * ps
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(len(n_parts)):
        for bj in range(len(m_parts)):
            for bk in range(len(p_parts)):
                ns = n_parts[bi]
                ms = m_parts[bj]
                ps = p_parts[bk]
                i0 = n_off[bi]
                j0 = m_off[bj]
                k0 = p_off[bk]
                
                if ns == 2 and ms == 2 and ps == 2:
                    i_list = [i0, i0 + 1]
                    j_list = [j0, j0 + 1]
                    k_list = [k0, k0 + 1]
                    col = _embed_strassen_block(U, V, W, col, i_list, j_list, k_list, n, m, p)
                else:
                    for i_s in range(ns):
                        for j_s in range(ms):
                            for k_s in range(ps):
                                i = i0 + i_s
                                j = j0 + j_s
                                k = k0 + k_s
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
    
    if col != rank:
        return None
    
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Define candidate partitions to try
    # Format: (n_parts, m_parts, p_parts)
    partitions = [
        # Current best: [1,2] x [2,1,2] x [2,1,2] -> rank 71
        ([1, 2], [2, 1, 2], [2, 1, 2]),
        # Try other partitions that might yield lower rank
        ([1, 2], [2, 2, 1], [2, 2, 1]),
        ([2, 1], [2, 2, 1], [2, 2, 1]),
        ([1, 1, 1], [2, 2, 1], [2, 2, 1]),
        ([1, 2], [3, 2], [3, 2]),
        ([2, 1], [3, 2], [3, 2]),
        ([1, 2], [2, 3], [2, 3]),
        ([2, 1], [2, 3], [2, 3]),
        ([3], [5], [5]),  # Pure schoolbook fallback
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for n_parts, m_parts, p_parts in partitions:
        if time.time() > deadline:
            break
        
        # Validate partition sums
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        
        result = _try_partition(n, m, p, n_parts, m_parts, p_parts)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    
    # Fallback to schoolbook if nothing works
    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
