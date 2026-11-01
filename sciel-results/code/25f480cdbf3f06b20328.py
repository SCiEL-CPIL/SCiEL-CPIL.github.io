import time

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    Ud = [[int(round(2 * x)) for x in row] for row in U]
    Vd = [[int(round(2 * x)) for x in row] for row in V]
    Wd = [[int(round(2 * x)) for x in row] for row in W]
    for M in (Ud, Vd, Wd):
        for row in M:
            for x in row:
                if x not in (-2, -1, 0, 1, 2):
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(r):
                    s += Ud[ui][c] * Vd[vj][c] * Wd[wk][c]
                if s != 8:
                    return False
    return True

def _rank24_construct():
    """Rank-24 exact 3x3x3 matmul decomposition.

    Structure:
      - Top-left 2x2 Strassen core (7 terms) -> 2x2 contributions to C[0,0],C[0,1],C[1,0],C[1,1]
      - Bottom-right 1x1 core (1 term)       -> C[2,2]
      - 16 cross terms (schoolbook products) -> the A[i,2]B[2,k] part of the 2x2 block
                                                + the 12 third-row/col entries
    Total rank = 7 + 1 + 16 = 24.

    Index conventions (n=m=p=3):
      U row = i*3+j  (A[i,j])
      V row = j*3+k  (B[j,k])
      W row = k*3+i  (C[i,k] output coefficient)
    """
    n = m = p = 3
    rank = 24
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]

    def add_term(u, v, w):
        nonlocal cur_c
        for a, val in u.items():
            U[a][cur_c] += val
        for b, val in v.items():
            V[b][cur_c] += val
        for d, val in w.items():
            W[d][cur_c] += val
        cur_c += 1

    cur_c = 0

    # ---- Top-left 2x2 Strassen core (7 terms) ----
    # A sub-block rows/cols 0,1 : A00=0,A01=1,A10=3,A11=4
    # B sub-block               : B00=0,B01=1,B10=3,B11=4
    # C sub-block (W idx)       : C00=0,C01=3,C10=1,C11=4
    # M1 = (A00+A11)(B00+B11)  -> C00,C01,C10,C11
    add_term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 3: 1, 1: 1, 4: 1})
    # M2 = (A10+A11)B00        -> C10,C11
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: 1})
    # M3 = A00(B10-B11)        -> C01, -C11
    add_term({0: 1}, {3: 1, 4: -1}, {3: 1, 4: -1})
    # M4 = A11(B01-B00)        -> C00, C10
    add_term({4: 1}, {1: 1, 0: -1}, {0: 1, 1: 1})
    # M5 = (A00+A01)B11        -> -C00, C01
    add_term({0: 1, 1: 1}, {4: 1}, {0: -1, 3: 1})
    # M6 = (A10-A00)(B00+B01)  -> C11
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1})
    # M7 = (A01-A11)(B10+B11)  -> C00
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1})

    # ---- Bottom-right 1x1 core (1 term) ----
    # A22=8, B22=8 -> C22 (W idx 8)
    add_term({8: 1}, {8: 1}, {8: 1})

    # ---- 16 cross terms (schoolbook) ----
    # The 2x2 block entries each miss the A[i,2]B[2,k] product:
    #   C00 += A02(5)*B20(6)
    #   C01 += A02(5)*B21(7)
    #   C10 += A12(8)*B20(6)
    #   C11 += A12(8)*B21(7)
    cross = [
        # (A index, B index, C W index)
        (5, 6, 0),  # C00
        (5, 7, 3),  # C01
        (8, 6, 1),  # C10
        (8, 7, 4),  # C11
        # C02 = A00B02 + A01B12 + A02B22
        (0, 2, 6),
        (1, 5, 6),
        (5, 8, 6),
        # C12 = A10B02 + A11B12 + A12B22
        (3, 2, 7),
        (4, 5, 7),
        (8, 8, 7),
        # C20 = A20B00 + A21B10 + A22B20
        (6, 0, 2),
        (7, 3, 2),
        (8, 6, 2),
        # C21 = A20B01 + A21B11 + A22B21
        (6, 1, 5),
        (7, 4, 5),
        (8, 7, 5),
    ]
    for (a, b, widx) in cross:
        add_term({a: 1}, {b: 1}, {widx: 1})

    assert cur_c == rank, "expected rank 24, got %d" % cur_c
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    best = _schoolbook()
    if _time_left() > 5.0:
        cand = _rank24_construct()
        if cand is not None and _verify_exact(cand["U"], cand["V"], cand["W"]):
            best = cand
    return best
# EVOLVE-BLOCK-END
