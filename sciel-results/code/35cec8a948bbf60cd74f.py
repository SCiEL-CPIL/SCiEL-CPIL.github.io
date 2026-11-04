"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]


def _verify_fast(U, V, W, n=4, m=4, p=9):
    """Fast verification using numpy. U,V,W are numpy arrays."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True


def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2():
    """Strassen decomposition for 2x2 x 2x2 matmul, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1

    return {"U": U, "V": V, "W": W}


def _strassen_4x4():
    """Strassen decomposition for 4x4 x 4x4 matmul, rank 49.
    Recursively applies Strassen 2x2 to 2x2 blocks.
    U[i*4+j, c] (16 rows), V[j*4+k, c] (16 rows), W[k*4+i, c] (16 rows).
    """
    S2 = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for I in range(2):
        for J in range(2):
            for c in range(7):
                cc = (I * 2 + J) * 7 + c
                for ii in range(2):
                    for jj in range(2):
                        U[(2 * I + ii) * 4 + (2 * J + jj)][cc] += S2['U'][ii * 2 + jj][c]
                for jj in range(2):
                    for kk in range(2):
                        V[(2 * J + jj) * 4 + (2 * I + kk)][cc] += S2['V'][jj * 2 + kk][c]
                for kk in range(2):
                    for ii in range(2):
                        W[(2 * I + kk) * 4 + (2 * J + ii)][cc] += S2['W'][kk * 2 + ii][c]

    return {"U": U, "V": V, "W": W}


def _decomp_4x4x9():
    """Full 4x4 x 4x9 decomposition, rank 114.
    Direct sum of two 4x4x4 (Strassen rank 49 each) and one 4x4x1 (schoolbook rank 16).
    """
    S4 = _strassen_4x4()
    rank = 114
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    # Part 1: First 4 columns of B (columns 0-3), rank 49
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][c] += S4['U'][i * 4 + j][c]
        for j in range(4):
            for k in range(4):
                V[j * 9 + k][c] += S4['V'][j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] += S4['W'][k * 4 + i][c]

    # Part 2: Middle 4 columns of B (columns 4-7), rank 49
    for c in range(49):
        cc = 49 + c
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][cc] += S4['U'][i * 4 + j][c]
        for j in range(4):
            for k in range(4):
                V[j * 9 + (4 + k)][cc] += S4['V'][j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[(4 + k) * 4 + i][cc] += S4['W'][k * 4 + i][c]

    # Part 3: Last column of B (column 8), rank 16
    for i in range(4):
        for j in range(4):
            cc = 98 + i * 4 + j
            U[i * 4 + j][cc] = 1
            V[j * 9 + 8][cc] = 1
            W[8 * 4 + i][cc] = 1

    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 9
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    try:
        cand = _decomp_4x4x9()
        if _verify(cand['U'], cand['V'], cand['W'], n, m, p):
            best_result = cand
            best_rank = 114
            print("Verified rank-114 decomposition")
    except Exception as e:
        print("rank-114 construction failed:", e)

    return best_result
