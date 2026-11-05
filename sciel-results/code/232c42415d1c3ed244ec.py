"""Winograd/Strassen block decomposition for 3x5x5 matrix multiplication."""
import time

DIMENSIONS = [3, 5, 5]

def _make_winograd_333():
    # F(2,3) Winograd for 3x3x3. r = 0..22, (i,k) = divmod(r,3).
    U = {}  # (i,j) -> coeff
    V = {}  # (j,k) -> coeff
    W = {}  # (i,k) -> coeff
    for r in range(23):
        i, k = divmod(r, 3)
        # U row i: (1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0)
        U[(i, 0)] = 1; U[(i, 1)] = 1; U[(i, 2)] = 1
        U[(i, 3)] = -1; U[(i, 4)] = -1
        U[(i, 9)] = 1; U[(i, 10)] = 1
        U[(i, 14)] = -1; U[(i, 15)] = -1
        U[(i, 19)] = 1; U[(i, 20)] = 1
        # V col k: (1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 0, -1, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0)
        V[(0, k)] = 1; V[(1, k)] = 1; V[(2, k)] = 1
        V[(6, k)] = -1; V[(7, k)] = -1; V[(8, k)] = -1
        V[(9, k)] = 1; V[(11, k)] = -1; V[(13, k)] = -1
    # W: row i, col k
    for i in range(3):
        for k in range(3):
            W[(i, k)] = 1
    return U, V, W

def _make_strassen_322():
    # Strassen for 3x2x2. r = 0..6.
    # A: 3x2 (rows i=0..2, cols j=0..1); B: 2x2 (rows j=0..1, cols k=0..1); C: 3x2
    U = {}  # (i,j)
    V = {}  # (j,k)
    W = {}  # (i,k)
    # r=0: M1
    U[(0,0)] = 1; U[(0,1)] = 1; U[(1,0)] = 1; U[(1,1)] = 1
    V[(0,0)] = 1; V[(1,0)] = 1
    W[(0,0)] = 1; W[(1,0)] = 1; W[(2,0)] = 1
    # r=1: M2
    U[(0,0)] = 1; U[(0,1)] = 1; U[(1,0)] = -1; U[(1,1)] = -1
    V[(0,1)] = 1; V[(1,1)] = 1
    W[(0,1)] = 1; W[(1,1)] = 1; W[(2,1)] = 1
    # r=2: M3
    U[(0,0)] = 1
    V[(0,0)] = 1; V[(0,1)] = -1
    W[(0,0)] = 1; W[(0,1)] = -1
    # r=3: M4
    U[(1,0)] = 1
    V[(1,0)] = 1; V[(1,1)] = -1
    W[(1,0)] = 1; W[(1,1)] = -1
    # r=4: M5
    U[(2,0)] = 1; U[(2,1)] = 1
    V[(0,0)] = 1
    W[(2,0)] = 1
    # r=5: M6
    U[(2,0)] = 1
    V[(1,1)] = 1
    W[(2,1)] = 1
    # r=6: M7
    U[(0,1)] = 1; U[(1,1)] = 1
    V[(1,1)] = 1; V[(0,1)] = -1
    W[(0,1)] = 1; W[(1,1)] = 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5
    # Block partition: m = 3 + 2, p = 3 + 2
    # Blocks (i-block, j-block, k-block):
    #   (0, 0, 0): 3x3x3 Winograd
    #   (0, 1, 0): 3x2x2 Strassen
    #   (0, 0, 1): 3x2x2 Strassen
    #   (0, 1, 1): 3x2x2 Strassen
    # (i-block is always 0 since n=3)

    total_rank = 23 + 3 * 7  # 44

    U = [[0] * total_rank for _ in range(n * m)]  # 15 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 15 x 44

    # Winograd block: i in [0,3), j in [0,3), k in [0,3)
    Uw, Vw, Ww = _make_winograd_333()
    for r in range(23):
        for (i, j), c in Uw.items():
            if c != 0:
                U[i * m + j][r] += c
        for (j, k), c in Vw.items():
            if c != 0:
                V[j * p + k][r] += c
        for (i, k), c in Ww.items():
            if c != 0:
                W[k * n + i][r] += c

    # Strassen blocks
    Us, Vs, Ws = _make_strassen_322()

    # Block (j in [3,5), k in [0,2))
    for r in range(7):
        rr = 23 + r
        for (i, j), c in Us.items():
            U[i * m + (j + 3)][rr] += c
        for (j, k), c in Vs.items():
            V[(j + 3) * p + k][rr] += c
        for (i, k), c in Ws.items():
            W[k * n + i][rr] += c

    # Block (j in [0,3), k in [2,4))
    for r in range(7):
        rr = 23 + 7 + r
        for (i, j), c in Us.items():
            U[i * m + j][rr] += c
        for (j, k), c in Vs.items():
            V[j * p + (k + 2)][rr] += c
        for (i, k), c in Ws.items():
            W[(k + 2) * n + i][rr] += c

    # Block (j in [3,5), k in [2,4))
    for r in range(7):
        rr = 23 + 14 + r
        for (i, j), c in Us.items():
            U[i * m + (j + 3)][rr] += c
        for (j, k), c in Vs.items():
            V[(j + 3) * p + (k + 2)][rr] += c
        for (i, k), c in Ws.items():
            W[(k + 2) * n + i][rr] += c

    if time.time() > deadline:
        # Should not happen; return what we have
        pass

    return {"U": U, "V": V, "W": W}
