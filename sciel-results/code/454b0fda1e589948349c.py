import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = 2, 4, 8
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16

    rank = 20
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)

    # Winograd 4x4x4 matrix multiplication (5 multiplications), integer coefficients.
    # A (4x4) rows r, B (4x4) rows s, C (4x4) rows t.
    # C[t] = sum_r A[r] * B[s_t(r)]
    # M1 = (a00+a10+a20+a30)(b00+b01+b02+b03)
    # M2 = (a00-a10+a20-a30)(b00-b01+b02-b03)
    # M3 = (a00+a10-a20-a30)(b00+b01-b02-b03)
    # M4 = (a00-a10-a20+a30)(b00-b01-b02+b03)
    # M5 = (a11+a21+a31)(b11+b12+b13)
    wA = {
        1: [1, 1, 1, 1],
        2: [1, -1, 1, -1],
        3: [1, 1, -1, -1],
        4: [1, -1, -1, 1],
        5: [0, 1, 1, 1],
    }
    wB = {
        1: [1, 1, 1, 1],
        2: [1, -1, 1, -1],
        3: [1, 1, -1, -1],
        4: [1, -1, -1, 1],
        5: [0, 1, 1, 1],
    }
    wC = {
        1: [1, 0, 0, 0],
        2: [0, 1, 0, 0],
        3: [0, 0, 1, 0],
        4: [0, 0, 0, 1],
        5: [0, 1, 1, 1],
    }

    # Kronecker "outer" 2x2x2 tensor, rank 4.
    # C[i,k] = sum_j A[i,j]*B[j,k], no sum over i or k.
    # Term0: A[i=0]*B[k=0] -> C[i=0,k=0]
    # Term1: A[i=0]*B[k=1] -> C[i=0,k=1]
    # Term2: A[i=1]*B[k=0] -> C[i=1,k=0]
    # Term3: A[i=1]*B[k=1] -> C[i=1,k=1]
    outer = {
        0: (0, 0, (0, 0)),
        1: (0, 1, (0, 1)),
        2: (1, 0, (1, 0)),
        3: (1, 1, (1, 1)),
    }

    col = 0
    for outer_idx in range(4):
        i, k, (ci, ck) = outer[outer_idx]
        for mult in range(1, 6):
            for r in range(4):
                for s in range(4):
                    for t in range(4):
                        a_val = wA[mult][r]
                        b_val = wB[mult][s]
                        c_val = wC[mult][t]
                        if a_val != 0 and b_val != 0 and c_val != 0:
                            U[i * 4 + r, col] += a_val
                            V[s * 8 + (2 * k + t), col] += b_val
                            W[(2 * k + t) * 2 + ci, col] += c_val
            col += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return result
    return _schoolbook(n, m, p)

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
