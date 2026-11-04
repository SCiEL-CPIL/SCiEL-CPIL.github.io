"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 130.0
DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n

    # Schoolbook decomposition (rank n*m*p), always valid
    R0 = n * m * p
    U0 = np.zeros((Nm, R0))
    V0 = np.zeros((Mp, R0))
    W0 = np.zeros((Pn, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]

    return {"U": _pack(U0), "V": _pack(V0), "W": _pack(W0)}
