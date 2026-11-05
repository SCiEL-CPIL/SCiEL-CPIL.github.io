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
    U = np.array([[1,0,1,0,1,-1,0],
                  [0,0,0,0,1,0,1],
                  [0,1,0,0,0,1,0],
                  [1,1,0,1,0,0,-1]], dtype=np.float64)
    V = np.array([[1,1,0,-1,0,1,0],
                  [0,0,1,0,0,1,0],
                  [0,0,0,1,0,0,1],
                  [1,0,-1,0,1,0,1]], dtype=np.float64)
    W = np.array([[1,0,0,1,-1,0,1],
                  [0,1,0,1,0,0,0],
                  [0,0,1,0,1,0,0],
                  [1,-1,1,0,0,1,0]], dtype=np.float64)
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
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
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


def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Verify decomposition using numpy for speed.
    Returns True if valid, False otherwise.
    """
    r = U.shape[1]
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Vectorize over c
    U_mat = U  # (n*m, r)
    V_mat = V  # (m*p, r)
    W_mat = W  # (p*n, r)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_mat[i*m+j, :]  # (r,)
                v_row = V_mat[j*p+k, :]  # (r,)
                w_row = W_mat[k*n+i, :]  # (r,)
                val = np.dot(np.dot(u_row, v_row), w_row)  # This is wrong!
                # Actually: sum_c u[c]*v[c]*w[c]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _generate_splits(dim, max_parts=3):
    """Generate all compositions of dim into up to max_parts positive integers."""
    splits = set()
    def helper(remaining, parts_so_far, max_parts):
        if len(parts_so_far) == max_parts:
            if remaining > 0:
                splits.add(tuple(parts_so_far + [remaining]))
            return
        for next_val in range(1, remaining + 1):
            helper(remaining - next_val, parts_so_far + [next_val], max_parts)
    helper(dim, [], max_parts)
    return list(splits)


def _estimate_rank(n_parts, m_parts, p_parts):
    """Estimate the rank for a given block decomposition."""
    total = 0
    for ni in n_parts:
        for mi in m_parts:
            for pi in p_parts:
                if ni == 2 and mi == 2 and pi == 2:
                    total += 7
                else:
                    total += ni * mi * pi
    return total


def construct():
    deadline = time.time() + 125  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Generate all possible splits
    n_splits = _generate_splits(n, max_parts=3)
    m_splits = _generate_splits(m, max_parts=3)
    p_splits = _generate_splits(p, max_parts=3)
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Generate all candidate combinations
    all_candidates = set()
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                all_candidates.add((tuple(ns), tuple(ms), tuple(ps)))
    
    # Sort by estimated rank
    sorted_candidates = sorted(all_candidates, key=lambda x: _estimate_rank(x[0], x[1], x[2]))
    
    # Try candidates in order of estimated rank
    for n_parts, m_parts, p_parts in sorted_candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, list(n_parts), list(m_parts), list(p_parts))
            if rank < best_rank:
                if _verify_decomposition_fast(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    # If we found rank 71 or better, we can stop early
                    if rank <= 71:
                        break
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
