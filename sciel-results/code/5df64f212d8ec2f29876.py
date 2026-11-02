import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    current_col = 0

    def add_strassen_block(k_start, k_end):
        nonlocal current_col
        k0 = k_start
        k1 = k_end
        
        # M1
        U[0][current_col] = 1
        U[3][current_col] = 1
        V[0*p + k0][current_col] = 1
        V[1*p + k1][current_col] = 1
        W[k0*n + 0][current_col] = 1
        W[k1*n + 1][current_col] = 1
        current_col += 1

        # M2
        U[2][current_col] = 1
        U[3][current_col] = 1
        V[0*p + k0][current_col] = 1
        W[k0*n + 1][current_col] = 1
        W[k1*n + 1][current_col] = -1
        current_col += 1

        # M3
        U[0][current_col] = 1
        V[0*p + k1][current_col] = 1
        V[1*p + k1][current_col] = -1
        W[k1*n + 0][current_col] = 1
        W[k1*n + 1][current_col] = 1
        current_col += 1

        # M4
        U[3][current_col] = 1
        V[1*p + k0][current_col] = 1
        V[0*p + k0][current_col] = -1
        W[k0*n + 0][current_col] = 1
        W[k0*n + 1][current_col] = 1
        current_col += 1

        # M5
        U[0][current_col] = 1
        U[1][current_col] = 1
        V[1*p + k1][current_col] = 1
        W[k0*n + 0][current_col] = -1
        W[k1*n + 0][current_col] = 1
        current_col += 1

        # M6
        U[2][current_col] = 1
        U[0][current_col] = -1
        V[0*p + k0][current_col] = 1
        V[0*p + k1][current_col] = 1
        W[k1*n + 1][current_col] = 1
        current_col += 1

        # M7
        U[1][current_col] = 1
        U[3][current_col] = -1
        V[1*p + k0][current_col] = 1
        V[1*p + k1][current_col] = 1
        W[k0*n + 0][current_col] = 1
        current_col += 1

    def add_trivial_col(k):
        nonlocal current_col
        U[0][current_col] = 1
        V[0*p + k][current_col] = 1
        W[k*n + 0][current_col] = 1
        current_col += 1

        U[1][current_col] = 1
        V[1*p + k][current_col] = 1
        W[k*n + 0][current_col] = 1
        current_col += 1

        U[2][current_col] = 1
        V[0*p + k][current_col] = 1
        W[k*n + 1][current_col] = 1
        current_col += 1

        U[3][current_col] = 1
        V[1*p + k][current_col] = 1
        W[k*n + 1][current_col] = 1
        current_col += 1

    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_strassen_block(6, 7)
    add_trivial_col(8)

    return {"U": U, "V": V, "W": W}
