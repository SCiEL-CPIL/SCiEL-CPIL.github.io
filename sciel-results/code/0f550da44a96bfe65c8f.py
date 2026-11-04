"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2 matrix multiplication."""
    # Each term: (a_coeffs, b_coeffs, c_coeffs) for 2x2 blocks
    # a_coeffs[i0*2+j0], b_coeffs[j0*2+k0], c_coeffs[k0*2+i0]
    terms = [
        ([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]),
        ([0,0, 1,1], [1,1, 0,0], [0,0, 1,-1]),
        ([1,1, 0,0], [0,0, 1,1], [0,1, 0,1]),
        ([0,0, 0,1], [0,-1, 0,1], [1,0, -1,0]),
        ([1,0, 0,0], [1,0, -1,0], [-1,1, 0,0]),
        ([-1,0, 1,0], [0,1, 0,1], [0,0, 0,1]),
        ([0,1, 0,-1], [1,0, 1,0], [1,0, 0,0]),
    ]
    return terms

def _build_decomposition_from_blocks(n, m, p, i_blocks, j_blocks, k_blocks):
    """
    Build a decomposition by tiling the matrix multiplication into blocks.
    For each triple (ib, jb, kb), use either Strassen (if 2x2x2) or schoolbook.
    Returns U, V, W, rank.
    """
    STRASSEN_TERMS = _strassen_2x2_terms()
    
    # Compute total rank
    total_rank = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                if ib == 2 and jb == 2 and kb == 2:
                    total_rank += 7
                else:
                    total_rank += ib * jb * kb
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                # Determine start indices
                i_start = sum(i_blocks[:i_blocks.index(ib)]) if ib in i_blocks else 0
                # Actually, we need to iterate with positions
                pass
    
    # Better: iterate with explicit positions
    col = 0
    i_pos = 0
    for ib_size in i_blocks:
        j_pos = 0
        for jb_size in j_blocks:
            k_pos = 0
            for kb_size in k_blocks:
                if ib_size == 2 and jb_size == 2 and kb_size == 2:
                    # Strassen
                    for t in range(7):
                        a_coeffs, b_coeffs, c_coeffs = STRASSEN_TERMS[t]
                        for di in range(2):
                            for dj in range(2):
                                idx_u = (i_pos + di) * m + (j_pos + dj)
                                U[idx_u][col] = a_coeffs[di * 2 + dj]
                        for dj in range(2):
                            for dk in range(2):
                                idx_v = (j_pos + dj) * p + (k_pos + dk)
                                V[idx_v][col] = b_coeffs[dj * 2 + dk]
                        for dk in range(2):
                            for di in range(2):
                                idx_w = (k_pos + dk) * n + (i_pos + di)
                                W[idx_w][col] = c_coeffs[dk * 2 + di]
                        col += 1
                else:
                    # Schoolbook
                    for di in range(ib_size):
                        for dj in range(jb_size):
                            for dk in range(kb_size):
                                idx_u = (i_pos + di) * m + (j_pos + dj)
                                idx_v = (j_pos + dj) * p + (k_pos + dk)
                                idx_w = (k_pos + dk) * n + (i_pos + di)
                                U[idx_u][col] = 1
                                V[idx_v][col] = 1
                                W[idx_w][col] = 1
                                col += 1
                k_pos += kb_size
            j_pos += jb_size
        i_pos += ib_size
    
    return U, V, W, col

def _all_partitions(dim):
    """Return all ways to partition a dimension into blocks of size 1 or 2."""
    results = []
    def _rec(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if remaining >= 2:
            current.append(2)
            _rec(remaining - 2, current)
            current.pop()
        if remaining >= 1:
            current.append(1)
            _rec(remaining - 1, current)
            current.pop()
    _rec(dim, [])
    return results

def _greedy_rank_reduce(U, V, W, n, m, p, max_iters=50, tol=1e-8):
    """
    Try to reduce rank by identifying redundant columns.
    A column c is redundant if removing it and adjusting others preserves the tensor.
    This is a heuristic: try zeroing out one column at a time and see if we can
    compensate with other columns.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Compute the full tensor
    T = np.zeros((n, m, p))
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    
    # Try to remove each column
    for c_remove in range(r):
        # Check if column c_remove is "zero" in some sense
        # A column is removable if its contribution can be absorbed
        # Simple heuristic: if the column has very small norm, remove it
        norm_u = np.linalg.norm(U[:, c_remove])
        norm_v = np.linalg.norm(V[:, c_remove])
        norm_w = np.linalg.norm(W[:, c_remove])
        if norm_u < tol or norm_v < tol or norm_w < tol:
            # Remove this column
            U = np.delete(U, c_remove, axis=1)
            V = np.delete(V, c_remove, axis=1)
            W = np.delete(W, c_remove, axis=1)
            r -= 1
    
    return U, V, W, r

def _try_merge_columns(U, V, W, n, m, p, max_attempts=100):
    """
    Try to merge pairs of columns into single columns.
    Two columns c1, c2 can be merged if their sum tensor is approximately rank-1.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    T = np.zeros((n, m, p))
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    
    merged = False
    for c1 in range(r):
        for c2 in range(c1 + 1, r):
            # Sum of two rank-1 tensors
            T_sum = np.einsum('i,j,k->ijk', U[:, c1], V[:, c1], W[:, c1]) + \
                    np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
            # Check if T_sum is approximately rank-1
            # Unfold and check singular values
            T_unfold = T_sum.reshape(n, m * p)
            try:
                s = np.linalg.svd(T_unfold, compute_uv=False)
                if len(s) > 1 and s[1] < 1e-6 * (s[0] + 1e-12):
                    # Approximately rank-1, merge
                    # Find best rank-1 approx
                    U_new, S_new, Vt_new = np.linalg.svd(T_unfold, full_matrices=False)
                    u_vec = U_new[:, 0] * S_new[0]
                    v_vec = Vt_new[0, :] * S_new[0]
                    # This doesn't directly give us the 3-way split
                    # Skip complex merging for now
                    pass
            except:
                pass
    
    return U, V, W, r

def _construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate all possible block partitions
    i_parts = _all_partitions(n)  # partitions of 3
    j_parts = _all_partitions(m)  # partitions of 4
    k_parts = _all_partitions(p)  # partitions of 8
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Enumerate all combinations of partitions
    for ip in i_parts:
        for jp in j_parts:
            for kp in k_parts:
                if time.time() > deadline - 30:
                    break
                try:
                    U, V, W, rank = _build_decomposition_from_blocks(n, m, p, ip, jp, kp)
                    if rank < best_rank:
                        # Verify
                        if _verify(U, V, W, n, m, p):
                            best_rank = rank
                            best_U = U
                            best_V = V
                            best_W = W
                except:
                    pass
    
    if best_U is not None:
        # Try greedy rank reduction
        U_arr, V_arr, W_arr, r = _greedy_rank_reduce(
            np.array(best_U, dtype=float), 
            np.array(best_V, dtype=float), 
            np.array(best_W, dtype=float), 
            n, m, p
        )
        if r < best_rank:
            # Verify reduced version
            if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p):
                return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
        
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    return _construct()
# EVOLVE-BLOCK-END
