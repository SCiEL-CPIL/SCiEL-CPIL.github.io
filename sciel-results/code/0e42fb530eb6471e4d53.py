"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0

    schoolbook_rank = n * m * p  # 80

    def build_schoolbook():
        rank = schoolbook_rank
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
        n, m, p = DIMENSIONS
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_strassen_4x4x4():
        """Build rank-49 Strassen decomposition for 4x4x4."""
        U2 = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ]
        V2 = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, -1, 0, 1, 0, 1],
        ]
        W2 = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0],
        ]

        rank = 49
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]

        A_meta = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 1, 1), (1, 1, -1)],
        ]

        B_meta = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
        ]

        C_meta = [
            [(0, 0, 1), (1, 1, 1), (1, 0, -1), (0, 1, 1)],
            [(0, 1, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, 1)],
        ]

        for c_meta in range(7):
            for c_sub in range(7):
                c = c_meta * 7 + c_sub
                for (bi, bj, coeff) in A_meta[c_meta]:
                    for i in range(2):
                        for j in range(2):
                            U[(bi*2+i)*2 + (bj*2+j)][c] += coeff * U2[i*2+j][c_sub]
                for (bj, bk, coeff) in B_meta[c_meta]:
                    for j in range(2):
                        for k in range(2):
                            V[(bj*2+j)*2 + (bk*2+k)][c] += coeff * V2[j*2+k][c_sub]
                for (ci, ck, coeff) in C_meta[c_meta]:
                    for i in range(2):
                        for k in range(2):
                            W[(ck*2+k)*2 + (ci*2+i)][c] += coeff * W2[k*2+i][c_sub]

        return {"U": U, "V": V, "W": W}

    def build_4x4x5():
        """Build 4x4x5: Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 4) = rank 53."""
        result_444 = build_strassen_4x4x4()
        U444 = result_444["U"]
        V444 = result_444["V"]
        W444 = result_444["W"]

        rank = 53
        U = [row[:] for row in U444]
        V = [row[:] for row in V444]
        W = [row[:] for row in W444]

        for i in range(16):
            U[i].extend([0]*4)
        for i in range(20):
            V[i].extend([0]*4)
        for i in range(20):
            W[i].extend([0]*4)

        for j in range(4):
            c = 49 + j
            U[j][c] = 1
            V[j*5+4][c] = 1
            for i in range(4):
                W[4*4+i][c] = 1

        return {"U": U, "V": V, "W": W}

    try:
        result = build_4x4x5()
        if verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass

    return build_schoolbook()
# EVOLVE-BLOCK-END
