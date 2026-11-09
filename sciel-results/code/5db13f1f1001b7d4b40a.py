import time
import numpy as np

DEADLINE = 145.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_seed_36():
    n, m, p = 2, 4, 5
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0

    # 7-term Strassen for a 2x2x2 block, as (U, V, W) sparse patterns.
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def add_strassen_block(i1, j1, k1):
        nonlocal col
        for u_pat, v_pat, w_pat in strassen_terms:
            for (di, dj), coeff in u_pat.items():
                U[(i1+di)*m + (j1+dj)][col] += coeff
            for (dj, dk), coeff in v_pat.items():
                V[(j1+dj)*p + (k1+dk)][col] += coeff
            for (di, dk), coeff in w_pat.items():
                W[(k1+dk)*n + (i1+di)][col] += coeff
            col += 1

    def add_schoolbook_2x2x1(i1, j1, k1):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                U[(i1+di)*m + (j1+dj)][col] = 1
                V[(j1+dj)*p + k1][col] = 1
                W[k1*n + (i1+di)][col] = 1
                col += 1

    # j in {0,1}x{2,3}, k in {0,1}x{2,3} -> 4 Strassen blocks (28)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    # k=4 handled schoolbook (8)
    add_schoolbook_2x2x1(0, 0, 4)
    add_schoolbook_2x2x1(0, 2, 4)
    return U, V, W

def _tensor_from_factors(U, V, W):
    n, m, p = 2, 4, 5
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for ri in range(r):
        T += np.einsum('a,b,c->abc', U[:, ri], V[:, ri], W[:, ri])
    return T

def _verify(U, V, W, tol=1e-9):
    n, m, p = 2, 4, 5
    T = _tensor_from_factors(U, V, W)
    Texp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[i*m+j, j*p+k, k*n+i] = 1.0
    return bool(np.allclose(T, Texp, atol=tol))

def _drop_cols(U, V, W, drop):
    keep = [r for r in range(len(U[0])) if r not in drop]
    U2 = [[row[r] for r in keep] for row in U]
    V2 = [[row[r] for r in keep] for row in V]
    W2 = [[row[r] for r in keep] for row in W]
    return U2, V2, W2

def _try_reduce(U, V, W, budget_s=110.0):
    n, m, p = 2, 4, 5
    end = START + budget_s
    Ua = [row[:] for row in U]
    Va = [row[:] for row in V]
    Wa = [row[:] for row in W]
    if not _verify(Ua, Va, Wa):
        return Ua, Va, Wa

    # Phase A: drop any all-zero columns.
    r = len(Ua[0])
    keep = [c for c in range(r)
            if any(Ua[i][c] != 0 for i in range(len(Ua)))
            or any(Va[i][c] != 0 for i in range(len(Va)))
            or any(Wa[i][c] != 0 for i in range(len(Wa)))]
    if len(keep) < r:
        Ua, Va, Wa = _drop_cols(Ua, Va, Wa, set(range(r)) - set(keep))

    # Phase B: iteratively try removing every single column (exhaustive).
    r = len(Ua[0])
    progress = True
    while progress and (time.time() - START) < end - 2.0:
        progress = False
        r = len(Ua[0])
        for c in range(r):
            if (time.time() - START) > end - 1.0:
                break
            U2, V2, W2 = _drop_cols(Ua, Va, Wa, {c})
            if _verify(U2, V2, W2):
                Ua, Va, Wa = U2, V2, W2
                progress = True
                break
    return Ua, Va, Wa

def construct():
    n, m, p = 2, 4, 5
    U, V, W = _build_seed_36()
    best = (len(U[0]), U, V, W)

    if _remaining() > 5.0:
        Ur, Vr, Wr = _try_reduce(best[1], best[2], best[3], budget_s=_remaining() - 4.0)
        if len(Ur[0]) < best[0] and _verify(Ur, Vr, Wr):
            best = (len(Ur[0]), Ur, Vr, Wr)

    U, V, W = best[1], best[2], best[3]
    return {"U": U, "V": W and W, "W": W} if False else {"U": U, "V": V, "W": W}
