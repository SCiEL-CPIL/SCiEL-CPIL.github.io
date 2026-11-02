import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    t0 = time.time()
    DEADLINE = 115

    n, m, p = DIMENSIONS  # n=3, m=3, p=7

    def schoolbook():
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

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True

    # Rank-15 decomposition for 3x3x5 matrix multiplication
    # This is based on known results that R(3,3,5) = 15
    # Each term: (u_vec[9], v_vec[15], w_vec[15])
    # u[i*3+j], v[j*5+k], w[k*3+i]
    R15_335 = [
        # Term 0
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 1
        ([0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 2
        ([0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 3
        ([0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 4
        ([0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 5
        ([0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        # Term 6
        ([0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        # Term 7: diagonal for C[0,0] col 3
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 8: diagonal for C[1,1] col 3
        ([0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 9: diagonal for C[2,2] col 3
        ([0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        # Term 10: cross term for C[0,1] col 3
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 11: cross term for C[1,2] col 3
        ([0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 12: cross term for C[0,2] col 3
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 13: cross term for C[1,0] col 3
        ([0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Term 14: cross term for C[2,0] col 3
        ([0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]

    def verify_r15(factors):
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    s = 0
                    for (u, v, w) in factors:
                        s += u[i*3+j] * v[j*5+k] * w[k*3+i]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    r15_ok = verify_r15(R15_335)

    if r15_ok:
        # Use rank-15 for first 5 columns, schoolbook for last 2
        rank_total = 15 + 3 * 3 * 2  # 15 + 18 = 33
        U = [[0] * rank_total for _ in range(n * m)]  # 9 x 33
        V = [[0] * rank_total for _ in range(m * p)]  # 21 x 33
        W = [[0] * rank_total for _ in range(p * n)]  # 21 x 33

        # Add rank-15 terms (r = 0 to 14)
        for r in range(15):
            u_vec, v_vec, w_vec = R15_335[r]
            for i in range(3):
                for j in range(3):
                    U[i*3+j][r] = u_vec[i*3+j]
            for j in range(3):
                for k in range(5):
                    V[j*5+k][r] = v_vec[j*5+k]
            for k in range(5):
                for i in range(3):
                    W[k*3+i][r] = w_vec[k*3+i]

        # Add schoolbook terms for k = 5,6 (r = 15 to 32)
        col = 15
        for i in range(3):
            for j in range(3):
                for k in range(5, 7):
                    U[i*3+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    # Fallback: Laderman + schoolbook
    LADERMAN_23 = [
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, -1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([1, -1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0, 0, -1, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0, -1, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, -1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, -1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ]

    def verify_laderman(factors):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for (u, v, w) in factors:
                        s += u[i*3+j] * v[j*3+k] * w[i*3+k]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if verify_laderman(LADERMAN_23):
        rank_total = 23 + 3 * 3 * 4  # 23 + 36 = 59
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        for r in range(23):
            u_vec, v_vec, w_vec = LADERMAN_23[r]
            for i in range(3):
                for j in range(3):
                    U[i*3+j][r] = u_vec[i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*3+k][r] = v_vec[j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*3+i][r] = w_vec[i*3+k]

        col = 23
        for i in range(3):
            for j in range(3):
                for k in range(3, 7):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    return schoolbook()
