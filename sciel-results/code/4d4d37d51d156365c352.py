import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
N, M, P = 2, 3, 6
DEADLINE = 110.0  # safety margin under 120s


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_222():
    """Return (U, V, W) for 2x2 @ 2x2 with rank 7 (Strassen)."""
    # Strassen's 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    U = [
        [1, 1, 1, 0, 1, 0, 0],  # a00
        [0, 0, 0, 1, 0, 1, -1], # a10
        [0, 1, 0, 0, 0, -1, 1], # a11
        [1, 0, 0, 0, 1, 0, 0],  # a01
    ]
    V = [
        [1, 0, 0, 1, 0, 1, 0],  # b00
        [0, 1, 0, 0, 1, 0, 1],  # b10
        [1, 0, 1, 0, 0, 0, 0],  # b01
        [0, 0, -1, -1, -1, 1, 1], # b11
    ]
    W = [
        [1, 0, -1, 1, 0, 0, 1],  # c00 (row 0 of C)
        [0, 1, 1, 0, 1, 0, 0],   # c10 (row 1 of C)
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [1, -1, 1, 0, 0, 1, 0],  # c11
    ]
    # Verify: for each output (i,j), sum_c U[i,j]*V[j,k]*W[k,i] should be 1
    # Actually the tensor is T[i,j,k] = 1 for all i,j,k in {0,1}
    # The decomposition: T = sum_c u_c * v_c * w_c where
    # u_c[i,j] = U[i][j] (U indexed by [i][j] for 2x2 matrix)
    # v_c[j,k] = V[j][k]
    # w_c[k,i] = W[k][i]
    return U, V, W


def _strassen_222():
    """Return (U, V, W) for 2x2 @ 2x2 with rank 7 (Strassen)."""
    U = [
        [1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, -1],
        [0, 1, 0, 0, 0, -1, 1],
        [1, 0, 0, 0, 1, 0, 0],
    ]
    V = [
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, 1, 1],
    ]
    W = [
        [1, 0, -1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W


def _strassen_212():
    """
    Rank-3 decomposition for 2x1 @ 1x2 (n=2, m=1, p=2).
    T[i, 0, k] = 1 for i in {0,1}, k in {0,1}.
    
    Decomposition: T[i,0,k] = sum_c U[i,c] * V[k,c] * W[k,i]
    where U: (2,3), V: (2,3) [p x rank], W: (2,2) [p x n]
    
    c0: U=(1,0), V=(1,1), W[:,0]=(1,1)
    c1: U=(0,1), V=(1,0), W[:,1]=(1,0)
    c2: U=(0,1), V=(0,1), W[:,2]=(0,1)
    """
    U = [
        [1, 0, 0],  # i=0
        [0, 1, 1],  # i=1
    ]
    V = [
        [1, 1, 0],  # k=0
        [1, 0, 1],  # k=1
    ]
    W = [
        [1, 1, 0],  # k=0: W[0][i] for i=0,1
        [1, 0, 1],  # k=1: W[1][i] for i=0,1
    ]
    return U, V, W


def _block_decomposition():
    """
    Decompose 2x3 @ 3x6 by partitioning m=3 into three blocks of size 1,
    and p=6 into three blocks of size 2.
    Each sub-problem is 2x1x2 with rank 3.
    Total rank: 9 * 3 = 27.
    """
    n, m, p = N, M, P
    m_blocks = [(0, 1), (1, 1), (2, 1)]
    p_blocks = [(0, 2), (2, 2), (4, 2)]
    
    all_U = []
    all_V = []
    all_W = []
    
    for (ms, msz) in m_blocks:
        for (ps, psz) in p_blocks:
            if msz == 1 and psz == 2:
                U_s, V_s, W_s = _strassen_212()
                for c in range(3):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    for i in range(n):
                        u[i * m + ms] = U_s[i][c]
                    v[ms * p + ps] = V_s[0][c]
                    v[ms * p + (ps + 1)] = 0.0  # V only has one column for m=1
                    for k in range(psz):
                        for i in range(n):
                            w[(ps + k) * n + i] = W_s[k][i]
                    all_U.append(u)
                    all_V.append(v)
                    all_W.append(w)
            elif msz == 2 and psz == 2:
                U_s, V_s, W_s = _strassen_222()
                for c in range(7):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    for i in range(n):
                        for jj in range(msz):
                            u[i * m + (ms + jj)] = U_s[i][jj]
                    for jj in range(msz):
                        for k in range(psz):
                            v[(ms + jj) * p + (ps + k)] = V_s[jj][k]
                    for k in range(psz):
                        for i in range(n):
                            w[(ps + k) * n + i] = W_s[k][i]
                    all_U.append(u)
                    all_V.append(v)
                    all_W.append(w)
            else:
                for i in range(n):
                    for jj in range(msz):
                        for k in range(psz):
                            u = [0.0] * (n * m)
                            v = [0.0] * (m * p)
                            w = [0.0] * (p * n)
                            u[i * m + (ms + jj)] = 1.0
                            v[(ms + jj) * p + (ps + k)] = 1.0
                            w[(ps + k) * n + i] = 1.0
                            all_U.append(u)
                            all_V.append(v)
                            all_W.append(w)
    
    rank = len(all_U)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c in range(rank):
        for r in range(n * m):
            U[r][c] = all_U[c][r]
        for r in range(m * p):
            V[r][c] = all_V[c][r]
        for r in range(p * n):
            W[r][c] = all_W[c][r]
    
    return {"U": U, "V": V, "W": W}


def _verify(res):
    U, V, W = res["U"], res["V"], res["W"]
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Try block decomposition (Strassen-based)
    res = None
    try:
        res = _block_decomposition()
    except Exception:
        res = None

    if res is not None and _verify(res):
        rank = len(res["U"][0])
        if rank < N * M * P:
            return res

    # Fallback schoolbook
    return _schoolbook()
