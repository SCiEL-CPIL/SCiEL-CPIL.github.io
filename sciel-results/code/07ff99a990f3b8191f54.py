import time

_DIMS = (3, 4, 6)
_T0 = time.time()
_DEADLINE = 120.0


def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 5.0)


def _build_1x4x2(terms, i_off, j_off, k_off):
    """
    Adds rank-1 terms for a 1x4x2 multiplication block.
    i: 0, j: 0..3, k: 0..1
    Rank: 8 (Schoolbook)
    """
    n, m, p = _DIMS
    for j in range(4):
        for k in range(2):
            # Term: A[i,j] * B[j,k] -> C[i,k]
            a_terms = [((i_off + 0), (j_off + j), 1)]
            b_terms = [((j_off + j), (k_off + k), 1)]
            c_terms = [((k_off + k), (i_off + 0), 1)]
            terms.append((a_terms, b_terms, c_terms))


def _build_2x2x2(terms, i_off, j_off, k_off):
    """
    Adds rank-1 terms for a 2x2x2 multiplication block using Strassen's algorithm.
    i: 0,1, j: 0,1, k: 0,1
    Rank: 7
    """
    # i_off, j_off, k_off are offsets into the global matrices
    
    # m1 = (a00 + a11) * (b00 + b11)
    terms.append((
        [((i_off + 0, j_off + 0, 1), (i_off + 1, j_off + 1, 1))],
        [((j_off + 0, k_off + 0, 1), (j_off + 1, k_off + 1, 1))],
        [((k_off + 0, i_off + 0, 1), (k_off + 1, i_off + 1, 1))]
    ))
    
    # m2 = (a10 + a11) * b00
    terms.append((
        [((i_off + 1, j_off + 0, 1), (i_off + 1, j_off + 1, 1))],
        [((j_off + 0, k_off + 0, 1))],
        [((k_off + 0, i_off + 1, 1), (k_off + 1, i_off + 1, -1))]
    ))
    
    # m3 = a00 * (b01 - b11)
    terms.append((
        [((i_off + 0, j_off + 0, 1))],
        [((j_off + 0, k_off + 1, 1), (j_off + 1, k_off + 1, -1))],
        [((k_off + 1, i_off + 0, 1), (k_off + 1, i_off + 1, 1))]
    ))
    
    # m4 = a11 * (b10 - b00)
    terms.append((
        [((i_off + 1, j_off + 1, 1))],
        [((j_off + 1, k_off + 0, 1), (j_off + 0, k_off + 0, -1))],
        [((k_off + 0, i_off + 0, 1), (k_off + 0, i_off + 1, 1))]
    ))
    
    # m5 = (a00 + a01) * b11
    terms.append((
        [((i_off + 0, j_off + 0, 1), (i_off + 0, j_off + 1, 1))],
        [((j_off + 1, k_off + 1, 1))],
        [((k_off + 0, i_off + 0, -1), (k_off + 1, i_off + 0, 1))]
    ))
    
    # m6 = (a10 - a00) * (b00 + b01)
    terms.append((
        [((i_off + 1, j_off + 0, 1), (i_off + 0, j_off + 0, -1))],
        [((j_off + 0, k_off + 0, 1), (j_off + 0, k_off + 1, 1))],
        [((k_off + 1, i_off + 1, 1))]
    ))
    
    # m7 = (a01 - a11) * (b10 + b11)
    terms.append((
        [((i_off + 0, j_off + 1, 1), (i_off + 1, j_off + 1, -1))],
        [((j_off + 1, k_off + 0, 1), (j_off + 1, k_off + 1, 1))],
        [((k_off + 0, i_off + 0, 1))]
    ))


def _build_3x4x2(terms, i_off, j_off, k_off):
    """
    Builds a 3x4x2 block.
    Splits i (3) into 1 + 2.
    Splits j (4) into 2 + 2.
    Rank: 8 (for 1x4x2) + 14 (for 2x4x2 which is 2x2x2 + 2x2x2) = 22.
    """
    # Block 1: i=0, j=0..3, k=0..1 -> 1x4x2
    _build_1x4x2(terms, i_off, j_off, k_off)
    
    # Block 2: i=1,2, j=0..1, k=0..1 -> 2x2x2
    _build_2x2x2(terms, i_off + 1, j_off, k_off)
    
    # Block 3: i=1,2, j=2..3, k=0..1 -> 2x2x2
    _build_2x2x2(terms, i_off + 1, j_off + 2, k_off)


def _gen_terms():
    """
    Generates terms for the full 3x4x6 multiplication.
    Splits k (6) into three blocks of 2.
    Total Rank: 3 * 22 = 66.
    """
    terms = []
    for ko in (0, 2, 4):
        _build_3x4x2(terms, 0, 0, ko)
    return terms


def _build_matrices(terms):
    n, m, p = _DIMS
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (a_terms, b_terms, c_terms) in enumerate(terms):
        for (i, j, c) in a_terms:
            U[i * m + j][r] += c
        for (j, k, c) in b_terms:
            V[j * p + k][r] += c
        for (k, i, c) in c_terms:
            W[k * n + i][r] += c
            
    return U, V, W


def _verify(U, V, W):
    """
    Verifies the tensor identity:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = _DIMS
    rank = len(U[0])
    
    # Optimization: Pre-compute partial sums or just iterate.
    # Given small dimensions, direct iteration is fine.
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_col = V[j * p + k]
                w_row = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += u_row[r] * v_col[r] * w_row[r]
                if s != 1:
                    return False
    return True


def _schoolbook():
    n, m, p = _DIMS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}


def construct():
    if not _deadline_ok():
        return _schoolbook()

    terms = _gen_terms()
    U, V, W = _build_matrices(terms)

    # Verify the decomposition is exact; fall back to schoolbook if not.
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback (should not trigger)
    return _schoolbook()
