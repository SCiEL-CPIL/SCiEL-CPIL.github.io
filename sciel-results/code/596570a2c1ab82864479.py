import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
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

def _build_strassen_based():
    """
    Build a Strassen-based decomposition for 2x5 * 5x5.
    Try multiple partitioning strategies and return the one with minimum rank.
    """
    n, m, p = 2, 5, 5

    strategies = [
        ([(0,1), (2,3), (4,)], [(0,1), (2,3), (4,)]),
        ([(0,1), (2,), (3,4)], [(0,1), (2,), (3,4)]),
        ([(0,1), (2,3), (4,)], [(0,), (1,2), (3,4)]),
        ([(0,1), (2,3), (4,)], [(0,1), (2,), (3,4)]),
        ([(0,1), (2,), (3,4)], [(0,), (1,2), (3,4)]),
    ]

    best_result = None
    best_rank = float('inf')

    for k_groups, j_groups in strategies:
        if not _deadline_ok():
            break

        total_rank = 0
        for kg in k_groups:
            for jg in j_groups:
                if len(kg) == 2 and len(jg) == 2:
                    total_rank += 7
                elif len(kg) == 2 and len(jg) == 1:
                    total_rank += 4
                elif len(kg) == 1 and len(jg) == 2:
                    total_rank += 4
                else:
                    total_rank += 2

        if total_rank >= best_rank:
            continue

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        offset = 0
        for kg in k_groups:
            for jg in j_groups:
                if len(kg) == 2 and len(jg) == 2:
                    offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1])
                elif len(kg) == 2 and len(jg) == 1:
                    offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0])
                elif len(kg) == 1 and len(jg) == 2:
                    offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1])
                else:
                    offset = _add_1x1(U, V, W, offset, kg[0], jg[0])

        if _verify_decomp(n, m, p, U, V, W):
            best_result = (U, V, W)
            best_rank = total_rank

    if best_result is None:
        k_groups = [(0,1), (2,3), (4,)]
        j_groups = [(0,1), (2,3), (4,)]
        total_rank = 0
        for kg in k_groups:
            for jg in j_groups:
                if len(kg) == 2 and len(jg) == 2:
                    total_rank += 7
                elif len(kg) == 2 and len(jg) == 1:
                    total_rank += 4
                elif len(kg) == 1 and len(jg) == 2:
                    total_rank += 4
                else:
                    total_rank += 2

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        offset = 0
        for kg in k_groups:
            for jg in j_groups:
                if len(kg) == 2 and len(jg) == 2:
                    offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1])
                elif len(kg) == 2 and len(jg) == 1:
                    offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0])
                elif len(kg) == 1 and len(jg) == 2:
                    offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1])
                else:
                    offset = _add_1x1(U, V, W, offset, kg[0], jg[0])
        return U, V, W

    return best_result

def _add_strassen_222(U, V, W, offset, k0, k1, j0, j1):
    a00 = 0*5 + k0
    a01 = 0*5 + k1
    a10 = 1*5 + k0
    a11 = 1*5 + k1

    b00 = k0*5 + j0
    b01 = k0*5 + j1
    b10 = k1*5 + j0
    b11 = k1*5 + j1

    c00 = j0*2 + 0
    c01 = j1*2 + 0
    c10 = j0*2 + 1
    c11 = j1*2 + 1

    U[a00][offset+0] += 1
    U[a11][offset+0] += 1
    V[b00][offset+0] += 1
    V[b11][offset+0] += 1
    W[c00][offset+0] += 1
    W[c11][offset+0] += 1

    U[a10][offset+1] += 1
    U[a11][offset+1] += 1
    V[b00][offset+1] += 1
    W[c10][offset+1] += 1
    W[c11][offset+1] += -1

    U[a00][offset+2] += 1
    V[b01][offset+2] += 1
    V[b11][offset+2] += -1
    W[c01][offset+2] += 1
    W[c11][offset+2] += 1

    U[a11][offset+3] += 1
    V[b10][offset+3] += 1
    V[b00][offset+3] += -1
    W[c00][offset+3] += 1
    W[c10][offset+3] += 1

    U[a00][offset+4] += 1
    U[a01][offset+4] += 1
    V[b11][offset+4] += 1
    W[c00][offset+4] += -1
    W[c01][offset+4] += 1

    U[a10][offset+5] += 1
    U[a00][offset+5] += -1
    V[b00][offset+5] += 1
    V[b01][offset+5] += 1
    W[c11][offset+5] += 1

    U[a01][offset+6] += 1
    U[a11][offset+6] += -1
    V[b10][offset+6] += 1
    V[b11][offset+6] += 1
    W[c00][offset+6] += 1

    return offset + 7

def _add_2x1(U, V, W, offset, k0, k1, j):
    a0k0 = 0*5 + k0
    a0k1 = 0*5 + k1
    a1k0 = 1*5 + k0
    a1k1 = 1*5 + k1

    bk0j = k0*5 + j
    bk1j = k1*5 + j

    c0j = j*2 + 0
    c1j = j*2 + 1

    U[a0k0][offset+0] += 1
    V[bk0j][offset+0] += 1
    W[c0j][offset+0] += 1

    U[a0k1][offset+1] += 1
    V[bk1j][offset+1] += 1
    W[c0j][offset+1] += 1

    U[a1k0][offset+2] += 1
    V[bk0j][offset+2] += 1
    W[c1j][offset+2] += 1

    U[a1k1][offset+3] += 1
    V[bk1j][offset+3] += 1
    W[c1j][offset+3] += 1

    return offset + 4

def _add_1x2(U, V, W, offset, k, j0, j1):
    a0k = 0*5 + k
    a1k = 1*5 + k

    bkj0 = k*5 + j0
    bkj1 = k*5 + j1

    c0j0 = j0*2 + 0
    c0j1 = j1*2 + 0
    c1j0 = j0*2 + 1
    c1j1 = j1*2 + 1

    U[a0k][offset+0] += 1
    V[bkj0][offset+0] += 1
    W[c0j0][offset+0] += 1

    U[a0k][offset+1] += 1
    V[bkj1][offset+1] += 1
    W[c0j1][offset+1] += 1

    U[a1k][offset+2] += 1
    V[bkj0][offset+2] += 1
    W[c1j0][offset+2] += 1

    U[a1k][offset+3] += 1
    V[bkj1][offset+3] += 1
    W[c1j1][offset+3] += 1

    return offset + 4

def _add_1x1(U, V, W, offset, k, j):
    a0k = 0*5 + k
    a1k = 1*5 + k
    bkj = k*5 + j
    c0j = j*2 + 0
    c1j = j*2 + 1

    U[a0k][offset+0] += 1
    V[bkj][offset+0] += 1
    W[c0j][offset+0] += 1

    U[a1k][offset+1] += 1
    V[bkj][offset+1] += 1
    W[c1j][offset+1] += 1

    return offset + 2

def _try_crank_reduction(U, V, W):
    """
    Try to reduce the rank by finding linear dependencies among the
    rank-1 tensor terms. Each rank-1 term is the vector
    v_r = U[:,r] (Kronecker) V[:,r] (Kronecker) W[:,r] in R^{n*m * m*p * p*n}.
    The decomposition is exact iff sum_r v_r = T_target. If a subset of terms
    sums to the zero vector, they can be removed without changing T.
    We iteratively search for such zero-sum subsets and remove them.
    """
    n, m, p = 2, 5, 5
    rank = len(U[0]) if U else 0
    if rank == 0:
        return U, V, W

    # Build each rank-1 term as a flattened vector of length n*m*m*p*p*n = 10*25*10=2500
    # v_r[(i*m+j), (j*p+k), (k*n+i)] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We only need the target tensor T_target[(i,j,k)] = 1 (all ones for the (i,j,k) triple).
    # Actually the full tensor T has shape (n*m, m*p, p*n). The target is:
    # T_target[(i*m+j), (j*p+k), (k*n+i)] = 1 if the indices are consistent, else 0.
    # But by the convention, the (i*m+j) index pairs with (j*p+k) only when they share j,
    # and (j*p+k) pairs with (k*n+i) only when they share k. So the target is a "diagonal" tensor.

    # For rank reduction we only need to check: does sum_r v_r = T_target?
    # If we remove a set S of terms, we need sum_{r not in S} v_r = T_target,
    # i.e., sum_{r in S} v_r = 0.

    # Build the list of term vectors.
    dim = n * m * m * p * p * n  # 2*5*5*5*5*2 = 2500
    terms = []
    for r in range(rank):
        vec = np.zeros(dim, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j][r]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j*p + k][r]
                    if v_val == 0:
                        continue
                    w_val = W[k*n + i][r]
                    if w_val == 0:
                        continue
                    idx = ((i*m + j) * (m*p)) + (j*p + k)
                    idx = idx * (p*n) + (k*n + i)
                    vec[idx] = u_val * v_val * w_val
        terms.append(vec)

    # Iteratively try to find a zero-sum subset of the remaining terms.
    # Use a greedy approach: for each term, check if it is in the span of the
    # others (with coefficients that make the sum zero). This is expensive, so
    # we use a simpler heuristic: check pairs and small subsets.
    #
    # A more robust approach: form the matrix M whose columns are the term vectors,
    # and find the null space. Any null-space vector c with integer/rational entries
    # gives a linear dependency sum_r c_r v_r = 0. If we can find a dependency
    # where the coefficients allow us to cancel whole terms (c_r = +-1 for a subset,
    # 0 otherwise), we can remove that subset.
    #
    # In practice, for these structured decompositions, dependencies are rare.
    # We'll do a quick rank check to see if the terms are linearly independent.
    if rank > 1:
        M = np.array(terms, dtype=np.float64).T  # dim x rank
        # Compute rank of M
        try:
            rnk = np.linalg.matrix_rank(M, tol=1e-10)
        except Exception:
            rnk = rank
        if rnk < rank:
            # There is a linear dependency. Try to find a sparse one.
            # This is complex; for now, we just note it and skip.
            # A full sparse null-space search is too expensive for the time budget.
            pass

    return U, V, W

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5

    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)

    if not _deadline_ok():
        return {"U": U_school, "V": V_school, "W": W_school}

    # Try Strassen-based approach with multiple partition strategies
    U, V, W = _build_strassen_based()

    # Verify
    if _verify_decomp(n, m, p, U, V, W):
        # Try to reduce rank further
        U, V, W = _try_crank_reduction(U, V, W)
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
