import numpy as np
import time
from fractions import Fraction

_START = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def construct():
    """
    Exact rank-66 tensor decomposition for (3x4)*(4x6) -> (3x6).
    Uses Strassen on rows (0,1) of each 3x2x2 sub-block + schoolbook on row 2.
    6 blocks * 11 terms = 66.
    """
    n, m, p = 3, 4, 6
    rank = 66
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0

    def add_block(jb, kb):
        nonlocal col
        # Strassen 2x2x2 on rows 0,1
        # A sub: indices (0,jb),(0,jb+1),(1,jb),(1,jb+1)
        # B sub: indices (jb,kb),(jb,kb+1),(jb+1,kb),(jb+1,kb+1)
        # C sub: (kb,0),(kb+1,0),(kb,1),(kb+1,1)
        a = 0 * m + jb
        b = 0 * m + jb + 1
        c = 1 * m + jb
        d = 1 * m + jb + 1
        e = jb * p + kb
        f = jb * p + kb + 1
        g = (jb + 1) * p + kb
        h = (jb + 1) * p + kb + 1
        p0 = kb * n + 0
        q0 = (kb + 1) * n + 0
        r0 = kb * n + 1
        s0 = (kb + 1) * n + 1

        # P1=(a+d)(e+h)
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[p0, col] = 1; W[s0, col] = 1
        col += 1
        # P2=(c+d)e
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[r0, col] = 1; W[s0, col] = -1
        col += 1
        # P3=a(f-h)
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[q0, col] = 1; W[s0, col] = 1
        col += 1
        # P4=d(g-e)
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[p0, col] = 1; W[r0, col] = 1
        col += 1
        # P5=(a+b)h
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[p0, col] = -1; W[q0, col] = 1
        col += 1
        # P6=(c-a)(e+f)
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[s0, col] = 1
        col += 1
        # P7=(b-d)(g+h)
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[p0, col] = 1
        col += 1

        # Schoolbook for row 2
        e2 = 2 * m + jb
        f2 = 2 * m + jb + 1
        t0 = kb * n + 2
        u0 = (kb + 1) * n + 2

        U[e2, col] = 1; V[e, col] = 1; W[t0, col] = 1; col += 1
        U[f2, col] = 1; V[g, col] = 1; W[t0, col] = 1; col += 1
        U[e2, col] = 1; V[f, col] = 1; W[u0, col] = 1; col += 1
        U[f2, col] = 1; V[h, col] = 1; W[u0, col] = 1; col += 1

    for jb in range(2):
        for kb in range(3):
            if _time_left() < 5.0:
                return _fallback()
            add_block(jb, kb)

    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vi = V[j * p + k]
                wi = W[k * n + i]
                total = sum(ui[r] * vi[r] * wi[r] for r in range(rank)
                            if ui[r] != 0 and vi[r] != 0 and wi[r] != 0)
                if total != 1:
                    return _fallback()

    # Convert to list of lists of plain ints
    U_out = [[int(x) for x in U[r]] for r in range(n * m)]
    V_out = [[int(x) for x in V[r]] for r in range(m * p)]
    W_out = [[int(x) for x in W[r]] for r in range(p * n)]
    return {"U": U_out, "V": V_out, "W": W_out}

def _fallback():
    n, m, p = 3, 4, 6
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
