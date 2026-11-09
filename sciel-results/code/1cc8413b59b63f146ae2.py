"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def _verify_fast(U, V, W, n, m, p, rank):
    """Fast verification using NumPy with vectorized operations."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all valid (i,j,k)
    # Vectorize: for each r, compute outer product contribution
    # T = sum_r U_col_r ⊗ V_col_r ⊗ W_col_r (with appropriate index mapping)
    # Let's do it semi-vectorized
    total = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_r = U_arr[:, r]  # shape (n*m,)
        v_r = V_arr[:, r]  # shape (m*p,)
        w_r = W_arr[:, r]  # shape (p*n,)
        # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        # Reshape u_r to (n, m), v_r to (m, p), w_r to (p, n)
        u_mat = u_r.reshape(n, m)
        v_mat = v_r.reshape(m, p)
        w_mat = w_r.reshape(p, n)
        # total[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is: for each (i,j,k), multiply three elements
        # Can do: np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        total += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
    return np.allclose(total, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
    """Schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _rank1_block(ni, nj, np_):
    """Rank-1 decomposition for any block where at least one dim is 1."""
    U = [[1] for _ in range(ni * nj)]
    V = [[1] for _ in range(nj * np_)]
    W = [[1] for _ in range(np_ * ni)]
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W

def _block_decomp(ni, nj, np_):
    """Get the best known decomposition for a block of size ni x nj x np_."""
    if ni == 1 or nj == 1 or np_ == 1:
        return _rank1_block(ni, nj, np_)
    if ni == 2 and nj == 2 and np_ == 2:
        return _strassen_2x2x2()
    # For 2x2xp or 2xpx2 or px2x2, we can potentially do better than schoolbook
    # But for now, use schoolbook as fallback
    return _schoolbook(ni, nj, np_)

def _block_rank(ni, nj, np_):
    """Get the rank for a block of size ni x nj x np_."""
    if ni == 1 or nj == 1 or np_ == 1:
        return 1
    if ni == 2 and nj == 2 and np_ == 2:
        return 7
    return ni * nj * np_

def _combine_decompositions(parts, n, m, p):
    """Combine a list of (U_part, V_part, W_part, offset_i, offset_j, offset_k, ni, nj, np)
    into a single decomposition for the full n x m x p problem."""
    total_rank = sum(len(U_part[0]) for U_part, _, _, _, _, _, _, _, _ in parts)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for U_part, V_part, W_part, off_i, off_j, off_k, ni, nj, np_ in parts:
        rank_part = len(U_part[0])
        for i in range(ni):
            for j in range(nj):
                a_full = (off_i + i) * m + (off_j + j)
                a_part = i * nj + j
                for r in range(rank_part):
                    U[a_full][col + r] = U_part[a_part][r]
        for j in range(nj):
            for k in range(np_):
                b_full = (off_j + j) * p + (off_k + k)
                b_part = j * np_ + k
                for r in range(rank_part):
                    V[b_full][col + r] = V_part[b_part][r]
        for k in range(np_):
            for i in range(ni):
                c_full = (off_k + k) * n + (off_i + i)
                c_part = k * ni + i
                for r in range(rank_part):
                    W[c_full][col + r] = W_part[c_part][r]
        col += rank_part
    
    return U, V, W

def _build_from_splits(n, m, p, i_sizes, j_sizes, k_sizes):
    """Build decomposition given specific split sizes for each dimension."""
    parts = []
    i_off = 0
    for ni in i_sizes:
        j_off = 0
        for nj in j_sizes:
            k_off = 0
            for np_ in k_sizes:
                U_part, V_part, W_part = _block_decomp(ni, nj, np_)
                parts.append((U_part, V_part, W_part, i_off, j_off, k_off, ni, nj, np_))
                k_off += np_
            j_off += nj
        i_off += ni
    
    U, V, W = _combine_decompositions(parts, n, m, p)
    return U, V, W

def _estimate_rank(i_sizes, j_sizes, k_sizes):
    """Estimate total rank for a given split."""
    total = 0
    for ni in i_sizes:
        for nj in j_sizes:
            for np_ in k_sizes:
                total += _block_rank(ni, nj, np_)
    return total

def _gen_splits(total, max_parts):
    """Generate all compositions of 'total' into at most 'max_parts' positive integers."""
    results = []
    def backtrack(remaining, parts):
        if not parts:
            if remaining > 0:
                results.append([remaining])
            return
        if len(parts) >= max_parts:
            results.append(list(parts))
            return
        for v in range(1, remaining + 1):
            parts.append(v)
            backtrack(remaining - v, parts)
            parts.pop()
    backtrack(total, [])
    seen = set()
    unique = []
    for s in results:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

def construct():
    start_time = _time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Generate all possible splits
    i_splits = _gen_splits(n, n)
    j_splits = _gen_splits(m, m)
    k_splits = _gen_splits(p, p)
    
    # Score each combination
    scored = []
    for isz in i_splits:
        for jsz in j_splits:
            for ksz in k_splits:
                total_r = _estimate_rank(isz, jsz, ksz)
                scored.append((total_r, isz, jsz, ksz))
    
    scored.sort(key=lambda x: x[0])
    
    # Try top candidates
    for total_r, isz, jsz, ksz in scored:
        if total_r >= best_rank:
            break
        if _time.time() > deadline:
            break
        U2, V2, W2 = _build_from_splits(n, m, p, isz, jsz, ksz)
        rank2 = len(U2[0]) if U2 else 0
        if rank2 < best_rank:
            if _verify_fast(U2, V2, W2, n, m, p, rank2):
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
                # Early exit if we hit a good rank
                if best_rank <= 50:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
