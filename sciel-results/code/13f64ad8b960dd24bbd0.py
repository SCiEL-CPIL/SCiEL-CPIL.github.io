"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time

def _build_decomposition(n, m, p):
    U = []
    V = []
    W = []

    def add_term(uvec, vvec, wvec):
        U.append([uvec.get((i, j), 0) for i in range(n) for j in range(m)])
        V.append([vvec.get((j, k), 0) for j in range(m) for k in range(p)])
        W.append([wvec.get((col, i), 0) for col in range(p) for i in range(n)])

    # Strassen for 2x2 @ 2x2, B columns c0,c1
    def strassen(c0, c1):
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        add_term({(0,0):1,(1,1):1}, {(0,c0):1,(0,c1):1,(1,c0):1,(1,c1):1}, {(c0,0):1,(c1,1):1})
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        add_term({(1,0):1,(1,1):1}, {(0,c0):1}, {(c0,1):1,(c1,1):-1})
        # M3 = A00(B10-B11) -> C01+=M3, C11+=M3
        add_term({(0,0):1}, {(1,c0):1,(1,c1):-1}, {(c1,0):1,(c1,1):1})
        # M4 = A11(B01-B10) -> C00+=M4, C10+=M4
        add_term({(1,1):1}, {(0,c1):1,(1,c0):-1}, {(c0,0):1,(c0,1):1})
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        add_term({(0,0):1,(0,1):1}, {(1,c1):1}, {(c0,0):-1,(c1,0):1})
        # M6 = (A10-A00)(B00+B10) -> C11+=M6
        add_term({(1,0):1,(0,0):-1}, {(0,c0):1,(1,c0):1}, {(c1,1):1})
        # M7 = (A01-A11)(B01+B11) -> C00+=M7
        add_term({(0,1):1,(1,1):-1}, {(0,c1):1,(1,c1):1}, {(c0,0):1})

    # 5 blocks: columns (0,1),(2,3),(4,5),(6,7),(8,9)
    for b in range(5):
        strassen(2*b, 2*b+1)

    # tail column 10: schoolbook 2 terms
    c = 10
    for j in range(m):
        add_term({(0,j):1,(1,j):1}, {(j,c):1}, {(c,0):1,(c,1):1})

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.sum(Ur[i, j, :] * Vr[j, k, :] * Wr[k, i, :]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                uvec = {(i,j):1}
                vvec = {(j,k):1}
                wvec = {(k,i):1}
                U.append([uvec.get((ii,jj),0) for ii in range(n) for jj in range(m)])
                V.append([vvec.get((jj,kk),0) for jj in range(m) for kk in range(p)])
                W.append([wvec.get((kk,ii),0) for kk in range(p) for ii in range(n)])
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    try:
        res = _build_decomposition(n, m, p)
        deadline = start + 120
        if time.time() < deadline - 15:
            if _verify(res["U"], res["V"], res["W"], n, m, p):
                return res
    except Exception:
        pass
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
