import random
from math import gcd
import time

n, m, p = 4, 4, 6

# Strassen 2x2 combination vectors (alpha, beta) for the four output quadrants.
# alpha pairs with A-sides, beta pairs with B-sides.
_STR22 = {
    0: [(1, 0), (0, 1)],          # C00 = M1 - M4 + M6 - M7
    1: [(0, 1), (1, 0)],          # C01 = M3 + M4
    2: [(1, 0), (1, 1)],          # C10 = M1 + M2
    3: [(1, 0), (-1, -1)],        # C11 = M2 - M3 + M5 - M7
}

# Strassen 2x1: A 2x2 times B 2x1, result 2x1.
#  M1=A00*B0, M2=(A00+A01)*B1, M3=(A10+A11)*B0, M4=A11*(B0+B1),
#  M5=(A10-A00)*(B0+B1), M6=(A01-A11)*(B0+B1)
#  C0 = M1 + M5 - M3,   C1 = M3 + M4
#  alpha pairs with A-sides, beta (scalar) multiplies the single B side.
_STR21 = {
    0: [(1, 1), (-1, -1), (-1, -1)],   # C0: M1 - M5 - M3  (M1=A00 B0)
    1: [(1, -1), (1, -1)],             # C1: M3 + M4        (M3,A10 B0; M4,A11 B1)
}


def _asym(A, B, C):
    out = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            row.append(sum(A[i][a] * B[b][j] for a in range(len(B)) for b in range(len(B))))
        out.append(row)
    for i in range(len(A)):
        for j in range(len(C)):
            assert out[i][j] == A[i][j] * C[j][0]
    return out


def _str22(A, B):
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
    ]


def _str21(A, B):
    return [A[0][0] * B[0] + A[0][1] * B[1], A[1][0] * B[0] + A[1][1] * B[1]]


def _check(A, B, C, name):
    nA = len(A)
    nB = len(B)
    m = nB
    for _ in range(300):
        a = [[random.randint(-5, 5) for _ in range(m)] for _ in range(nA)]
        b = [[random.randint(-5, 5) for _ in range(nB)] for _ in range(m)]
        exp = _asym(a, b, C)
        got = _str22(a, b) if name == '22' else _str21(a, b)
        if name == '22':
            for i in range(2):
                for j in range(2):
                    assert got[i][j] == exp[i][j], (name, a, b, i, j)
        else:
            for i in range(2):
                assert got[i] == exp[i], (name, a, b, i)


def _build_2x2():
    rows = []
    for c in range(4):
        for t in range(1, 8):
            (as_, ab), (bs, bb) = _STR22[c][t - 1]
            if bs is None:
                as_, bs, ab, bb = 0, 0, 1, 0
            rows.append((as_, bs, ab, bb))
    return rows


def _build_4x4x4():
    rows = []
    for r in range(4):          # top-block rows 0..3
        rows.append((r // 2, r % 2))
    for r in range(2):          # bottom-block rows 4..5 (i.e. row 2,3)
        rows.append((1, r))
    return rows


def _build_4x4x2():
    rows = []
    for r in range(4):          # A rows 0..3, block 0/1
        rows.append((r // 2, r % 2))
    for r in range(2):          # B rows 0,1 -> block 0
        rows.append((0, r))
    for r in range(2):          # B rows 2,3 -> block 1
        rows.append((1, r))
    return rows


def _str21_terms():
    terms = []
    for c in (0, 1):
        for (as_, bs) in _STR21[c]:
            if c == 0 and bs == 1:
                terms.append((as_, 0))    # B0
            elif c == 0:
                terms.append((as_, 1))    # B1 (M5)
            else:
                terms.append((as_, 1 if bs == -1 else 0))  # M3(A10)->0, M4(A11)->1
    return terms


def _assemble():
    U, V, W = [], [], []

    # --- 4x4x4 part: 49 terms ---
    a_rows = _build_4x4x4()
    b_rows = _build_4x4x4()
    for a0 in range(2):
        for a1 in range(2):
            for (as_, bs, ab, bb) in _build_2x2():
                U.append(1 if (a0, a1) == (as_, bs) else 0)
                V.append(1 if (a0, a1) == (ab, bb) else 0)
                W.append((a0, a1))     # (rowA, rowB) -> (i, k)

    # --- 4x4x2 part: 28 terms ---
    a_rows2 = _build_4x4x2()
    b_rows2 = _build_4x4x2()
    terms = _str21_terms()
    for (as_, bs) in terms:
        U.append(1 if (a0, a1) == (as_, bs) else 0)
        V.append(1 if (a0, a1) == (as_, bs) else 0)
        W.append((a0, a1))
    return U, V, W


def construct():
    deadline = time.time() + 115

    U, V, W = _assemble()
    rank = len(U)
    assert rank == 77

    # Verify against random integer matrices.
    for _ in range(200):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for t in range(rank):
                    if U[i * m + k][t] and V[k * n + i][t]:
                        acc += W[k * n + i][t]
                # expected A*B at (i,k)
                e = sum(A[i][j] * B[j][k] for j in range(m))
                assert acc == e, (i, k, acc, e)
                if time.time() > deadline:
                    break

    # Reduce by gcd per column (keeps exact decomposition).
    for t in range(rank):
        g = 0
        for r in range(n * m):
            g = gcd(g, U[r][t])
        for r in range(m * p):
            g = gcd(g, V[r][t])
        for r in range(p * n):
            g = gcd(g, W[r][t])
        if g > 1:
            for r in range(n * m):
                U[r][t] //= g
            for r in range(m * p):
                V[r][t] //= g
            for r in range(p * n):
                W[r][t] //= g

    return {"U": U, "V": V, "W": W}
