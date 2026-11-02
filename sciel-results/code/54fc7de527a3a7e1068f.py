"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def construct():
    if time.time() > _DEADLINE:
        return _schoolbook()
    try:
        result = _construct_rank28()
        if result is not None:
            return result
    except Exception:
        pass
    if time.time() > _DEADLINE:
        return _schoolbook()
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_rank28():
    n, m, p = DIMENSIONS
    assert n == 2 and m == 4 and p == 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: -1}),
    ]

    P1 = [0, 1, 4, 5]
    Q1 = [0, 1, 4, 5]
    R1 = [0, 2, 1, 3]
    P2 = [0, 1, 4, 5]
    Q2 = [2, 3, 6, 7]
    R2 = [4, 6, 5, 7]
    P3 = [2, 3, 6, 7]
    Q3 = [8, 9, 12, 13]
    R3 = [0, 2, 1, 3]
    P4 = [2, 3, 6, 7]
    Q4 = [10, 11, 14, 15]
    R4 = [4, 6, 5, 7]

    subs = [(P1, Q1, R1), (P2, Q2, R2), (P3, Q3, R3), (P4, Q4, R4)]

    col = 0
    for P_map, Q_map, R_map in subs:
        for P_f, Q_f, R_f in strassen_terms:
            for loc, coeff in P_f.items():
                U[P_map[loc]][col] += coeff
            for loc, coeff in Q_f.items():
                V[Q_map[loc]][col] += coeff
            for loc, coeff in R_f.items():
                W[R_map[loc]][col] += coeff
            col += 1

    if not _verify(U, V, W, n, m, p):
        return None
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
