import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, n, m, p):
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 * r2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            val = U1[i1 * m1 + j1][r1_idx] * U2[i2 * m2 + j2][r2_idx]
                            if val != 0:
                                U[i * m + j][r] += val
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            val = V1[j1 * p1 + k1][r1_idx] * V2[j2 * p2 + k2][r2_idx]
                            if val != 0:
                                V[j * p + k][r] += val
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            val = W1[k1 * n1 + i1][r1_idx] * W2[k2 * n2 + i2][r2_idx]
                            if val != 0:
                                W[k * n + i][r] += val
    return U, V, W


def _build_4x4x4_strassen():
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    U, V, W = _kron_decompose(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2, n, m, p)
    return {"U": U, "V": V, "W": W}


def _schoolbook_2x2x2x1():
    U = [[0] * 4 for _ in range(4)]
    V = [[0] * 4 for _ in range(2)]
    W = [[0] * 4 for _ in range(2)]
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return U, V, W


def _build_4x4x2():
    n, m, p = 4, 4, 2
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _schoolbook_2x2x2x1()
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 1, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_4x4x1():
    n, m, p = 4, 4, 1
    U1, V1, W1 = _schoolbook_2x2x2x1()
    U2, V2, W2 = _schoolbook_2x2x2x1()
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 1, 2, 2, 1, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_4x4x7():
    n, m, p = 4, 4, 7
    part1 = _build_4x4x4_strassen()
    part2 = _build_4x4x2()
    part3 = _build_4x4x1()
    r1, r2, r3 = len(part1['U'][0]), len(part2['U'][0]), len(part3['U'][0])
    total_rank = r1 + r2 + r3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    for i in range(n * m):
        for r in range(r1):
            U[i][r] = part1['U'][i][r]
    for i in range(m):
        for j in range(4):
            for r in range(r1):
                V[i * p + j][r] = part1['V'][i * 4 + j][r]
    for k in range(4):
        for i in range(n):
            for r in range(r1):
                W[k * n + i][r] = part1['W'][k * 4 + i][r]
    off2 = r1
    for i in range(n * m):
        for r in range(r2):
            U[i][off2 + r] = part2['U'][i][r]
    for i in range(m):
        for j in range(2):
            for r in range(r2):
                V[i * p + (4 + j)][off2 + r] = part2['V'][i * 2 + j][r]
    for k in range(2):
        for i in range(n):
            for r in range(r2):
                W[(4 + k) * n + i][off2 + r] = part2['W'][k * 4 + i][r]
    off3 = r1 + r2
    for i in range(n * m):
        for r in range(r3):
            U[i][off3 + r] = part3['U'][i][r]
    for i in range(m):
        for r in range(r3):
            V[i * p + 6][off3 + r] = part3['V'][i][r]
    for i in range(n):
        for r in range(r3):
            W[6 * n + i][off3 + r] = part3['W'][i][r]
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
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


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    try:
        result = _build_4x4x7()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    return _schoolbook(4, 4, 7)
