"""Rewritten program: rank-28 Strassen-based decomposition for (2,4,4) matmul."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

DEADLINE = 150.0
_START = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, trials=200, seed=12345):
    """Check that sum_r (U[:,r]·A)(V[:,r]·B) W[:,r] == A·B for random A,B."""
    rng = np.random.default_rng(seed)
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        C_true = A @ B
        C_rec = np.zeros((n, p))
        for r in range(rank):
            a = U[:, r] @ A.reshape(-1)
            b = V[:, r] @ B.reshape(-1)
            if a == 0 or b == 0:
                continue
            cvec = (a * b) * W[:, r]
            # W index k*n+i -> C[i,k]
            for k in range(p):
                for i in range(n):
                    C_rec[i, k] += cvec[k * n + i]
        if not np.allclose(C_true, C_rec, atol=1e-6):
            return False
    return True

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

def _strassen_244(n, m, p):
    # n=2, m=4, p=4
    # Strassen (2,2,2) terms: each (A_coeffs, B_coeffs, C_coeffs) over 4 entries row-major.
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # p2
        ([1,0,0,0], [0,-1,1,0], [0,1,0,0]),  # p3
        ([0,0,0,1], [-1,1,0,0], [1,0,0,0]),  # p4
        ([1,1,0,0], [0,0,0,1], [-1,0,0,-1]), # p5
        ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),  # p6
        ([1,0,0,-1], [0,1,0,1], [0,1,1,0]),  # p7
    ]
    # 4 sub-problems: (A_map, B_map, C_map) each a list of 4 global indices for local [00,01,10,11]
    subs = [
        ([0,1,4,5],   [0,1,4,5],   [0,2,1,3]),   # A1, B[0:2,0:2], C[:,0:2]
        ([0,1,4,5],   [2,3,6,7],   [4,6,5,7]),   # A1, B[0:2,2:4], C[:,2:4]
        ([2,3,6,7],   [8,9,12,13], [0,2,1,3]),   # A2, B[2:4,0:2], C[:,0:2]
        ([2,3,6,7],   [10,11,14,15],[4,6,5,7]),  # A2, B[2:4,2:4], C[:,2:4]
    ]
    rank = 4 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for Amap, Bmap, Cmap in subs:
        for Ac, Bc, Cc in S:
            for l in range(4):
                U[Amap[l]][col] += Ac[l]
                V[Bmap[l]][col] += Bc[l]
                W[Cmap[l]][col] += Cc[l]
            col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    if n != 2 or m != 4 or p != 4:
        return _schoolbook(n, m, p)
    try:
        cand = _strassen_244(n, m, p)
        if _time_remaining() > 5:
            if _verify(cand["U"], cand["V"], cand["W"], n, m, p, trials=300):
                return cand
    except Exception:
        pass
    return _schoolbook(n, m, p)
