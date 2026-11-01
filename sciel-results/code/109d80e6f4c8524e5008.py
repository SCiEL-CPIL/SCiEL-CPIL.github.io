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

def _rank26_construct():
    """Rank-26 construction: 2x2 Strassen (7 terms) + 19 cross terms."""
    n = m = p = 3
    R = 26
    U = [[0.0] * R for _ in range(9)]
    V = [[0.0] * R for _ in range(9)]
    W = [[0.0] * R for _ in range(9)]
    
    terms = [
        # 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (inner part)
        ({0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1, 1:1, 4:1}),
        ({3:1, 4:1}, {0:1}, {1:1, 4:1}),
        ({0:1}, {3:1, 4:-1}, {3:1, 4:-1}),
        ({4:1}, {1:1, 0:-1}, {0:1, 1:1}),
        ({0:1, 1:1}, {4:1}, {0:-1, 3:1}),
        ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),
        ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),
        # Cross terms for C[0:2,0:2] missing A[*,2]B[2,*] contributions
        ({2:1}, {6:1}, {0:1}),   # A02*B20 -> C00
        ({2:1}, {7:1}, {3:1}),   # A02*B21 -> C01
        ({5:1}, {6:1}, {1:1}),   # A12*B20 -> C10
        ({5:1}, {7:1}, {4:1}),   # A12*B21 -> C11
        # C[0,2] = A00B02 + A01B12 + A02B22
        ({0:1}, {2:1}, {6:1}),
        ({1:1}, {5:1}, {6:1}),
        ({2:1}, {8:1}, {6:1}),
        # C[1,2] = A10B02 + A11B12 + A12B22
        ({3:1}, {2:1}, {7:1}),
        ({4:1}, {5:1}, {7:1}),
        ({5:1}, {8:1}, {7:1}),
        # C[2,0] = A20B00 + A21B10 + A22B20
        ({6:1}, {0:1}, {3:1}),
        ({7:1}, {3:1}, {3:1}),
        ({8:1}, {6:1}, {3:1}),
        # C[2,1] = A20B01 + A21B11 + A22B21
        ({6:1}, {1:1}, {5:1}),
        ({7:1}, {4:1}, {5:1}),
        ({8:1}, {7:1}, {5:1}),
        # C[2,2] = A20B02 + A21B12 + A22B22
        ({6:1}, {2:1}, {8:1}),
        ({7:1}, {5:1}, {8:1}),
        ({8:1}, {8:1}, {8:1}),
    ]
    
    assert len(terms) == R, f"Expected {R} terms, got {len(terms)}"
    
    for c, (ud, vd, wd) in enumerate(terms):
        for idx, val in ud.items():
            U[idx][c] += val
        for idx, val in vd.items():
            V[idx][c] += val
        for idx, val in wd.items():
            W[idx][c] += val
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    best = _schoolbook()
    if _time_left() > 5.0:
        cand = _rank26_construct()
        if _verify_exact(cand["U"], cand["V"], cand["W"]):
            best = cand
    return best
# EVOLVE-BLOCK-END
