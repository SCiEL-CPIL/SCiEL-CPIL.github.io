import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]
_START = time.time()
_DEADLINE = 105.0

def _left():
    return _DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, R):
    # exact check with Fraction
    # convert to Fraction
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for k in range(p):
            for i1 in range(n):
                for j1 in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            s = Fraction(0)
                            for r in range(R):
                                s += Wf[k*n+i][r] * Uf[i1*m+j1][r] * Vf[j2*p+k2][r]
                            target = 1 if (i1==i and k2==k and j1==j2) else 0
                            if s != target:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # schoolbook
    R0 = n*m*p
    U = [[0]*R0 for _ in range(n*m)]
    V = [[0]*R0 for _ in range(m*p)]
    W = [[0]*R0 for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = R0
    # search for lower rank (best-effort, time-bounded)
    # ... (attempt) ...
    return best
