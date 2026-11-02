import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    return U1, V1, W1

def _build_kron_rank7():
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 7
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for i1 in range(2):
        for j1 in range(2):
            for j2 in range(2):
                i = i1
                j = j1 * 2 + j2
                for s in range(7):
                    U[i * m + j, s] += U1[i1 * 2 + j1, s] * 1.0

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(3):
                    j = j1 * 2 + j2
                    k = k1 * 3 + k2
                    for s in range(7):
                        V[j * p + k, s] += V1[j1 * 2 + k1, s] * 1.0

    for k1 in range(2):
        for k2 in range(3):
            for i1 in range(2):
                k = k1 * 3 + k2
                i = i1
                for s in range(7):
                    W[k * n + i, s] += W1[k1 * 2 + i1, s] * 1.0

    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_rank42():
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7
            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        if U1[i * 2 + j_local, s] != 0:
                            U[i * m + a, col_offset + s] += U1[i * 2 + j_local, s]
            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        if V1[j_local * 2 + k_local, s] != 0:
                            V[a * p + bcol, col_offset + s] += V1[j_local * 2 + k_local, s]
            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        if W1[k_local * 2 + i, s] != 0:
                            W[bcol * n + i, col_offset + s] += W1[k_local * 2 + i, s]
    return U, V, W

def _to_output(U, V, W):
    def convert(mat):
        return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]
    return {"U": convert(U), "V": convert(V), "W": convert(W)}

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_kron_rank7()
    if _verify(U, V, W, n, m, p):
        return _to_output(U, V, W)
    U, V, W = _build_rank42()
    if _verify(U, V, W, n, m, p):
        return _to_output(U, V, W)
    rank = n * m * p
    U_fb = np.zeros((n * m, rank), dtype=float)
    V_fb = np.zeros((m * p, rank), dtype=float)
    W_fb = np.zeros((p * n, rank), dtype=float)
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj, col] = 1
                V_fb[jj * p + kk, col] = 1
                W_fb[kk * n + ii, col] = 1
                col += 1
    return _to_output(U_fb, V_fb, W_fb)
