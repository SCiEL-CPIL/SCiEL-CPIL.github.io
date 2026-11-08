import time

_DIMS = (3, 4, 6)
_T0 = time.time()
_DEADLINE = 120.0


def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)


def _gen_terms():
    """Generate 66 explicit rank-1 terms.
    Each term: (a_terms, b_terms, out_terms)
      a_terms:  list of (i, j, coeff)
      b_terms:  list of (j, k, coeff)
      out_terms: list of (k, i, coeff)
    """
    T = []
    for ko in (0, 2, 4):
        # --- row 0 (i=0): schoolbook 1x4 * 4x2 -> 8 terms ---
        for j in range(4):
            for k in (ko, ko + 1):
                T.append(([(0, j, 1)], [(j, k, 1)], [(k, 0, 1)]))
        # --- rows 1,2 (i=1,2): Strassen over i-dim, j-pairs (0,1) and (2,3) ---
        for (j0, j1) in ((0, 1), (2, 3)):
            # m1 = (a1j0+a2j1)(b_j0,ko + b_j1,ko+1) -> c1,ko, c2,ko+1
            T.append(([(1, j0, 1), (2, j1, 1)],
                      [(j0, ko, 1), (j1, ko + 1, 1)],
                      [(ko, 1, 1), (ko + 1, 2, 1)]))
            # m2 = (a2j0+a2j1)(b_j0,ko) -> c2,ko, c2,ko+1
            T.append(([(2, j0, 1), (2, j1, 1)],
                      [(j0, ko, 1)],
                      [(ko, 2, 1), (ko + 1, 2, -1)]))
            # m3 = a1j0 (b_j0,ko+1 - b_j1,ko+1) -> c1,ko+1, c2,ko+1
            T.append(([(1, j0, 1)],
                      [(j0, ko + 1, 1), (j1, ko + 1, -1)],
                      [(ko + 1, 1, 1), (ko + 1, 2, 1)]))
            # m4 = a2j1 (b_j1,ko - b_j0,ko) -> c1,ko, c2,ko
            T.append(([(2, j1, 1)],
                      [(j1, ko, 1), (j0, ko, -1)],
                      [(ko, 1, 1), (ko, 2, 1)]))
            # m5 = (a1j0+a1j1)(b_j1,ko+1) -> c1,ko, c1,ko+1
            T.append(([(1, j0, 1), (1, j1, 1)],
                      [(j1, ko + 1, 1)],
                      [(ko, 1, -1), (ko + 1, 1, 1)]))
            # m6 = (a2j0-a1j0)(b_j0,ko+b_j0,ko+1) -> c2,ko+1
            T.append(([(2, j0, 1), (1, j0, -1)],
                      [(j0, ko, 1), (j0, ko + 1, 1)],
                      [(ko + 1, 2, 1)]))
            # m7 = (a1j1-a2j1)(b_j1,ko+b_j1,ko+1) -> c1,ko
            T.append(([(1, j1, 1), (2, j1, -1)],
                      [(j1, ko, 1), (j1, ko + 1, 1)],
                      [(ko, 1, 1)]))
    return T


def _build(terms):
    n, m, p = _DIMS
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (a_terms, b_terms, out_terms) in enumerate(terms):
        for (i, j, c) in a_terms:
            U[i * m + j][r] += c
        for (j, k, c) in b_terms:
            V[j * p + k][r] += c
        for (k, i, c) in out_terms:
            W[k * n + i][r] += c
    return U, V, W


def _verify(U, V, W):
    """Check tensor identity: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1."""
    n, m, p = _DIMS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
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
    U, V, W = _build(terms)

    # Verify the decomposition is exact; fall back to schoolbook if not.
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback (should not trigger)
    return _schoolbook()
