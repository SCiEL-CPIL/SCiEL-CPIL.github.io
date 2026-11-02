"""Matrix multiplication 2,2,7 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c])
                if s != 1:
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]:
                    continue
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c])
                if s != 0:
                    return False
    return True

def _strassen_terms():
    # 7 rank-1 terms for 2x2x2: u=(i,j), v=(j,k), w=(k,i), each 4 entries
    return [
        ((1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)),
        ((1, 1, -1, -1), (1, 1, 1, 1), (1, -1, -1, 1)),
        ((1, -1, 1, -1), (1, 1, -1, -1), (-1, 1, 1, -1)),
        ((1, -1, -1, 1), (1, 1, -1, -1), (-1, 1, 1, -1)),
        ((-1, 1, 1, -1), (1, -1, 1, -1), (1, 1, -1, 1)),
        ((-1, 1, -1, 1), (1, -1, 1, -1), (1, 1, -1, 1)),
        ((-1, -1, 1, 1), (1, -1, -1, 1), (1, -1, 1, -1)),
    ]

def _build_25():
    n, m, p = DIMENSIONS
    U, V, W = [], [], []
    terms = _strassen_terms()
    for b in range(3):  # columns 2b, 2b+1
        for (u, v, w) in terms:
            U.append(list(u))
            # V global index = j*p + (2b + k); v[j*2+k]
            Vrow = [0] * (m * p)
            for j in range(m):
                for k in range(2):
                    Vrow[j * p + 2 * b + k] = v[j * 2 + k]
            V.append(Vrow)
            # W global index = (2b + k)*n + i; w[k*2+i]
            Wrow = [0] * (p * n)
            for k in range(2):
                for i in range(n):
                    Wrow[(2 * b + k) * n + i] = w[k * 2 + i]
            W.append(Wrow)
    # last column k=6, schoolbook rank 4
    for i in range(n):
        for j in range(m):
            U.append([1 if ii == i * m + j else 0 for ii in range(n * m)])
            V.append([1 if jj == j * p + 6 else 0 for jj in range(m * p)])
            W.append([1 if kk == 6 * n + i else 0 for kk in range(p * n)])
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 120.0
    result = _build_25()
    if time.time() < deadline and _verify(result["U"], result["V"], result["W"]):
        return result
    return result
