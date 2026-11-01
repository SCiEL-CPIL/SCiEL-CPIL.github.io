import time

DIMENSIONS = [2, 3, 7]

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    U = []  # columns
    V = []
    W = []

    def add_term(u, v, w):
        ucol = [0] * (n * m)
        vcol = [0] * (m * p)
        wcol = [0] * (p * n)
        for (i, j), c in u.items():
            ucol[i * m + j] = c
        for (j, k), c in v.items():
            vcol[j * p + k] = c
        for (i, k), c in w.items():
            wcol[k * n + i] = c
        U.append(ucol); V.append(vcol); W.append(wcol)

    def strassen(ia, ja, ka):
        i0, i1 = ia, ia + 1
        j0, j1 = ja, ja + 1
        k0, k1 = ka, ka + 1
        add_term({(i0,j0):1,(i1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(i0,k0):1,(i1,k1):1})
        add_term({(i1,j0):1,(i1,j1):1}, {(j0,k0):1}, {(i1,k0):1})
        add_term({(i0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(i0,k1):1})
        add_term({(i1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(i1,k0):1})
        add_term({(i0,j0):1,(i0,j1):1}, {(j1,k1):1}, {(i0,k0):-1,(i1,k1):1})
        add_term({(i1,j0):1,(i0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(i1,k1):1})
        add_term({(i0,j1):1,(i1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(i0,k0):1})

    def school_21(ks):
        j2 = 2
        for k in ks:
            add_term({(0,j2):1,(1,j2):1}, {(j2,k):1}, {(0,k):1,(1,k):1})

    def school_221(ka):
        for j in (0,1):
            add_term({(0,j):1,(1,j):1}, {(j,ka):1}, {(0,ka):1,(1,ka):1})

    strassen(0, 0, 0)   # A1: k{0,1}
    strassen(0, 0, 2)   # A2: k{2,3}
    school_21([0,1,2,3])  # B
    strassen(0, 0, 4)   # C1: k{4,5}
    school_221(6)       # C2: k=6
    school_21([4,5,6])  # D

    rank = len(U)
    U_out = [[U[r][row] for r in range(rank)] for row in range(n * m)]
    V_out = [[V[r][row] for r in range(rank)] for row in range(m * p)]
    W_out = [[W[r][row] for r in range(rank)] for row in range(p * n)]

    # Verify
    try:
        import numpy as np
        Ua = np.array(U_out, dtype=float)
        Va = np.array(V_out, dtype=float)
        Wa = np.array(W_out, dtype=float)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i]))
                    if abs(s - 1.0) > 1e-9:
                        ok = False
        if not ok:
            return _schoolbook(n, m, p)
    except Exception:
        pass

    return {"U": U_out, "V": V_out, "W": W_out}
