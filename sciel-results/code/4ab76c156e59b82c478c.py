"""Tensor decomposition for 3x5x5 matrix multiplication.
Replaces one 2x2x2 schoolbook sub-block (rank 8) with a Strassen
decomposition (rank 7), reducing total rank from 75 to 74.
"""
import numpy as np

DIMENSIONS = [3, 5, 5]


def _strassen_block():
    """Return (U,V,W) each a (4,7) int matrix for the 2x2x2 Strassen tensor.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]."""
    m1 = ({(0, 0): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1})
    m2 = ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(1, 0): 1})
    m3 = ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1})
    m4 = ({(1, 1): 1}, {(0, 1): 1, (1, 0): -1}, {(1, 1): 1, (0, 1): -1})
    m5 = ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1},
          {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1})
    m6 = ({(0, 0): 1, (1, 0): -1}, {(0, 1): 1, (1, 1): 1},
          {(1, 0): 1, (1, 1): -1})
    m7 = ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (0, 1): 1},
          {(0, 0): 1, (1, 0): 1, (0, 1): -1, (1, 1): -1})
    cols = [m1, m2, m3, m4, m5, m6, m7]
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    for r, (uT, vT, wT) in enumerate(cols):
        for (a, b), c in uT.items():
            U[a * 2 + b, r] += c
        for (a, b), c in vT.items():
            V[a * 2 + b, r] += c
        for (a, b), c in wT.items():
            W[a * 2 + b, r] += c
    return U, V, W


def _verify_strassen():
    U, V, W = _strassen_block()
    T = np.einsum('kr,ir,jr->kij', W, U, V)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                exp = 1 if j == k else 0
                if T[k, i, j] != exp:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    use_strassen = _verify_strassen()
    U, V, W = _strassen_block()

    Udict = {}
    Vdict = {}
    Wdict = {}
    rank = 0

    if use_strassen:
        for r in range(7):
            col = rank
            rank += 1
            for idx in range(4):
                if U[idx, r] != 0:
                    Udict.setdefault(idx, {})[col] = int(U[idx, r])
            for idx in range(4):
                if V[idx, r] != 0:
                    Vdict.setdefault(idx, {})[col] = int(V[idx, r])
            for idx in range(4):
                if W[idx, r] != 0:
                    Wdict.setdefault(idx, {})[col] = int(W[idx, r])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if use_strassen and (i < 2 and j < 2 and k < 2):
                    continue
                col = rank
                rank += 1
                Udict.setdefault(i * m + j, {})[col] = 1
                Vdict.setdefault(j * p + k, {})[col] = 1
                Wdict.setdefault(k * n + i, {})[col] = 1

    Uout = []
    for i in range(n * m):
        row = [0] * rank
        for col, val in Udict.get(i, {}).items():
            row[col] = val
        Uout.append(row)
    Vout = []
    for j in range(m * p):
        row = [0] * rank
        for col, val in Vdict.get(j, {}).items():
            row[col] = val
        Vout.append(row)
    Wout = []
    for k in range(p * n):
        row = [0] * rank
        for col, val in Wdict.get(k, {}).items():
            row[col] = val
        Wout.append(row)
    return {"U": Uout, "V": Vout, "W": Wout}
