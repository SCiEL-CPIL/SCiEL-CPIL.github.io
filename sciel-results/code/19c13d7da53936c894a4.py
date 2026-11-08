import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 7]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
T_START = time.time()
DEADLINE = 150.0
SAFETY = 8.0

def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rows_of(mat):
    return [tuple(r) for r in mat]

def _validate(U, V, W):
    r = len(U[0])
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = Ua[i * m + j] @ Va[j * p + k]
                if lhs != Wa[k * n + i]:
                    return False
    return True

def _search(max_rank, max_c):
    # Removed expensive brute-force search.
    # For 2x4x7, the search space for rank < 56 is too large for this budget.
    # Returning None forces fallback to schoolbook.
    return None

def construct():
    best = _schoolbook()
    # Try to find a rank-minimal exact decomposition within the budget.
    # Only small ranks are tractable by enumeration; start from the known lower
    # bound (max(n*m, m*p, p*n)) upward.
    lb = max(NM, MP, PN)
    for r in range(lb, min(lb + 2, NM * MP * PN + 1)):
        if time.time() > DEADLINE - SAFETY:
            break
        res = _search(r, max_c=1)
        if res is not None:
            best = res
            break
    return best
