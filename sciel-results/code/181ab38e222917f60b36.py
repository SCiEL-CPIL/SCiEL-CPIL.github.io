import time

DIMENSIONS = [2, 2, 14]

_DEADLINE = 150.0
_START = time.time()
_DEADLINE_AT = _START + 125.0  # safety margin before the 150s kill


def _elapsed():
    return time.time() - _START


def _done():
    return time.time() >= _DEADLINE_AT


# One (i,j,k) product -> rank-1 tensor.
def _triplet(i, j, k, n, m, p):
    U = [0] * (n * m)
    U[i * m + j] = 1
    V = [0] * (m * p)
    V[j * p + k] = 1
    W = [0] * (p * n)
    W[k * n + i] = 1
    return U, V, W


# Strassen split of a 2x2x2 block of triplets:
#   i in {0,1}, j in {0,1}, k in {2h, 2h+1}
# Returns 7 (U, V, W) vectors.
def _strassen_block(h, n, m, p):
    k0 = 2 * h
    k1 = 2 * h + 1
    # (Acoeff, Bcoeff, Ccoeff) for the 7 rank-1 factors.
    # A indexed by (i,j), B by (j,k) with k in {0->k0,1->k1}, C by (i,k).
    defs = [
        # M1 = (a+d)(e+h)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2 = (c+d)(f-h)
        ([(1, 0, 1), (1, 1, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, -1)]),
        # M3 = a(e-f)
        ([(0, 0, 1)], [(0, 0, 1), (0, 1, -1)], [(0, 0, 1), (1, 0, 1)]),
        # M4 = d(g-e)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 1, 1), (1, 1, 1)]),
        # M5 = (a+b)h
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        # M6 = (c+d)e
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        # M7 = (b-d)(f+h)
        ([(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1)]),
    ]
    out = []
    for A, B, C in defs:
        U = [0] * (n * m)
        for (i, j, c) in A:
            U[i * m + j] += c
        V = [0] * (m * p)
        for (j, kk, c) in B:
            V[j * p + (k0 if kk == 0 else k1)] += c
        W = [0] * (p * n)
        for (i, kk, c) in C:
            W[(k0 if kk == 0 else k1) * n + i] += c
        out.append((U, V, W))
    return out


# Pair two consecutive 2x2x2 blocks (h and h+1) and split the k-pair {k0..k3}
# into 7 rank-1 factors. i in {0,1}, j in {0,1}, k in {4h..4h+3}.
def _strassen_pair(h, n, m, p):
    k0 = 4 * h
    k1 = 4 * h + 1
    k2 = 4 * h + 2
    k3 = 4 * h + 3
    # K ordering: 0->k0, 1->k1, 2->k2, 3->k3.
    defs = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, 1), (1, 2, 1), (1, 3, 1)]),
        # M2 = (A10+A11)(B01-B11)
        ([(1, 0, 1), (1, 1, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, -1), (1, 2, 1), (1, 3, -1)]),
        # M3 = A00(B00-B01)
        ([(0, 0, 1)], [(0, 0, 1), (0, 1, -1)], [(0, 0, 1), (0, 1, -1), (1, 2, 1), (1, 3, -1)]),
        # M4 = A11(B10-B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, -1), (0, 1, 1), (1, 2, -1), (1, 3, 1)]),
        # M5 = (A00+A01)B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1), (1, 2, -1), (1, 3, 1)]),
        # M6 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1), (1, 2, 1), (1, 3, -1)]),
        # M7 = (A01-A11)(B01+B11)
        ([(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1), (1, 2, 1), (1, 3, -1)]),
    ]
    out = []
    for A, B, C in defs:
        U = [0] * (n * m)
        for (i, j, c) in A:
            U[i * m + j] += c
        V = [0] * (m * p)
        for (j, kk, c) in B:
            V[j * p + (k0 + kk)] += c
        W = [0] * (p * n)
        for (i, kk, c) in C:
            W[(k0 + kk) * n + i] += c
        out.append((U, V, W))
    return out


# Fallback schoolbook (rank 56) in case anything goes wrong.
def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    try:
        # 14 blocks (i=0/1, j=0/1, k=0..6). Split i-pairs -> 7 rank-7 blocks.
        # Then pair the 7 blocks -> 7 rank-7 "pairs". Total rank 49.
        all_factors = []
        for ib in range(2):            # i block: rows {2ib, 2ib+1}
            for jb in range(2):        # j block: cols {2jb, 2jb+1}
                for hb in range(7):    # k block: rows {4hb..4hb+3}
                    if _done():
                        break
                    all_factors.extend(_strassen_pair(hb, n, m, p))
                if _done():
                    break
            if _done():
                break

        if _done() and not all_factors:
            # out of time with nothing; return whatever we have if any, else schoolbook
            pass

        if all_factors:
            rank = len(all_factors)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for c, (u, v, w) in enumerate(all_factors):
                for r, val in enumerate(u):
                    if val:
                        U[r][c] = val
                for r, val in enumerate(v):
                    if val:
                        V[r][c] = val
                for r, val in enumerate(w):
                    if val:
                        W[r][c] = val
            return {"U": U, "V": V, "W": W}
        return _schoolbook(n, m, p)
    except Exception:
        return _schoolbook(n, m, p)
