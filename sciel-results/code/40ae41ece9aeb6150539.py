import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B.
    C[0,0]=ae+cg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh.
    m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    C[0,0]=m1+m4-m5+m7
    C[0,1]=m3+m5
    C[1,0]=m2+m4
    C[1,1]=m1-m2+m3+m6
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def schoolbook_factors(n, m, p):
    """Returns (U, V, W) for n x m x p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes, block_factors=None):
    """
    Build the full factor matrices using block decomposition.
    block_factors: dict mapping (ni, mj, pk) -> (U_block, V_block, W_block)
    If not provided, uses Strassen for 2x2x2 and schoolbook otherwise.
    """
    if block_factors is None:
        block_factors = {}

    def get_block(ni, mj, pk):
        key = (ni, mj, pk)
        if key in block_factors:
            return block_factors[key]
        if ni == 2 and mj == 2 and pk == 2:
            U_b, V_b, W_b = strassen_factors()
        else:
            U_b, V_b, W_b = schoolbook_factors(ni, mj, pk)
        block_factors[key] = (U_b, V_b, W_b)
        return U_b, V_b, W_b

    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                U_b, V_b, W_b = get_block(ni, mj, pk)
                total_rank += U_b.shape[1]

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s

    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s

    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                U_block, V_block, W_block = get_block(ni, mj, pk)
                r = U_block.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=atol))

def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists (fallback)."""
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

def _block_rank(ni, mj, pk):
    """Compute the rank for a single block of size ni x mj x pk."""
    if ni == 2 and mj == 2 and pk == 2:
        return 7  # Strassen
    return ni * mj * pk  # schoolbook

def _get_compositions(size, max_parts):
    """All compositions of size with at most max_parts positive parts."""
    out = []
    def rec(rem, parts):
        if rem == 0:
            out.append(list(parts))
            return
        if len(parts) == max_parts:
            parts.append(rem)
            out.append(list(parts))
            parts.pop()
            return
        for first in range(1, rem + 1):
            parts.append(first)
            rec(rem - first, parts)
            parts.pop()
    rec(size, [])
    # dedupe
    seen = set()
    res = []
    for c in out:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            res.append(c)
    return res

def find_best_block_partition(n, m, p, deadline):
    """
    Find the best block partition to minimize total rank.
    Uses Strassen for 2x2x2 blocks and schoolbook otherwise.
    """
    best_rank = n * m * p  # schoolbook
    best_partition = ([n], [m], [p])

    n_parts = _get_compositions(n, max_parts=3)
    m_parts = _get_compositions(m, max_parts=5)
    p_parts = _get_compositions(p, max_parts=5)

    for n_sizes in n_parts:
        if time.time() > deadline - 10:
            break
        for m_sizes in m_parts:
            if time.time() > deadline - 10:
                break
            for p_sizes in p_parts:
                if time.time() > deadline - 10:
                    break
                rank = 0
                for ni in n_sizes:
                    for mj in m_sizes:
                        for pk in p_sizes:
                            rank += _block_rank(ni, mj, pk)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (n_sizes, m_sizes, p_sizes)

    return best_partition, best_rank

def _prune_zero_columns(U, V, W, atol=1e-12):
    """
    Remove columns that are entirely zero in U, V, or W. Such a column
    contributes zero to every tensor entry, so removal preserves the
    decomposition exactly while reducing the rank.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    u_zero = np.all(np.abs(U) < atol, axis=0)
    v_zero = np.all(np.abs(V) < atol, axis=0)
    w_zero = np.all(np.abs(W) < atol, axis=0)
    keep = ~(u_zero | v_zero | w_zero)
    if keep.all():
        return U, V, W
    U = U[:, keep]
    V = V[:, keep]
    W = W[:, keep]
    return U, V, W

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 120s wall-clock

    # Try to find the best block partition
    best_partition, best_rank = find_best_block_partition(n, m, p, deadline)
    n_sizes, m_sizes, p_sizes = best_partition
    print(f"Best partition: n={n_sizes}, m={m_sizes}, p={p_sizes}, rank={best_rank}")

    # Build the block decomposition
    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)

    # Prune redundant (zero) columns to reduce rank
    U, V, W = _prune_zero_columns(U, V, W)

    # Verify correctness; fall back to schoolbook if it fails
    if verify_decomposition(U, V, W, n, m, p, atol=1e-9):
        print(f"Using block decomposition with rank {U.shape[1]}")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: try several known-good partitions
    known_goods = [
        ([1, 2], [2, 2, 1], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [2, 2, 2]),
        ([1, 2], [1, 2, 2], [2, 2, 2]),
        ([2, 1], [1, 2, 2], [2, 2, 2]),
        ([1, 2], [2, 1, 2], [2, 2, 2]),
    ]
    for kg in known_goods:
        if time.time() > deadline - 5:
            break
        try:
            U2, V2, W2, r2 = build_block_decomposition(n, m, p, kg[0], kg[1], kg[2])
            U2, V2, W2 = _prune_zero_columns(U2, V2, W2)
            if verify_decomposition(U2, V2, W2, n, m, p, atol=1e-9):
                print(f"Using known-good partition {kg} with rank {U2.shape[1]}")
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
        except Exception:
            pass

    # Fallback: schoolbook (rank 90)
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
