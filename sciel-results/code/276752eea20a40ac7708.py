"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # Block decomposition over the 15 output columns.
    # Leave column 0 as a trivial single column (rank 4).
    # Pair the remaining 14 columns (1..14) into 7 Strassen blocks (rank 7 each).
    # Total rank = 4 + 7 * 7 = 53.

    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # Strassen for 2x2 * 2x2 over columns (k0, k1):
    #   U rows: 0=a, 1=b, 2=c, 3=d
    #   V rows: k0=B[0,k0], k1=B[0,k1], p+k0=B[1,k0], p+k1=B[1,k1]
    #   W rows: 2k0=C[0,k0], 2k0+1=C[1,k0], 2k1=C[0,k1], 2k1+1=C[1,k1]
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        # M1 = (a + d)(B[0,k0] + B[1,k1])
        U[0][r1] = 1
        U[3][r1] = 1
        V[k0][r1] = 1
        V[p + k1][r1] = 1

        # M2 = (c + d) * B[0,k0]
        U[2][r2] = 1
        U[3][r2] = 1
        V[k0][r2] = 1

        # M3 = a * (B[0,k1] - B[1,k1])
        U[0][r3] = 1
        V[k1][r3] = 1
        V[p + k1][r3] = -1

        # M4 = d * (B[1,k0] - B[0,k0])
        U[3][r4] = 1
        V[p + k0][r4] = 1
        V[k0][r4] = -1

        # M5 = (a + b) * B[1,k1]
        U[0][r5] = 1
        U[1][r5] = 1
        V[p + k1][r5] = 1

        # M6 = (c - a) * (B[0,k0] + B[0,k1])
        U[2][r6] = 1
        U[0][r6] = -1
        V[k0][r6] = 1
        V[k1][r6] = 1

        # M7 = (b - d) * (B[1,k0] + B[1,k1])
        U[1][r7] = 1
        U[3][r7] = -1
        V[p + k0][r7] = 1
        V[p + k1][r7] = 1

        # C[0, k0] = M1 + M4 - M5 + M7
        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1

        # C[1, k0] = M2 + M4
        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1

        # C[0, k1] = M3 + M5
        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1

        # C[1, k1] = M1 - M2 + M3 + M6
        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1

    # Trivial block for column 0 (rank 4), placed first.
    base_r = 0
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3

    # C[0, 0] = a*B[0,0] + b*B[1,0]
    U[0][r1] = 1
    V[0][r1] = 1
    W[0][r1] = 1

    U[1][r2] = 1
    V[p + 0][r2] = 1
    W[0][r2] = 1

    # C[1, 0] = c*B[0,0] + d*B[1,0]
    U[2][r3] = 1
    V[0][r3] = 1
    W[1][r3] = 1

    U[3][r4] = 1
    V[p + 0][r4] = 1
    W[1][r4] = 1

    # Strassen blocks over column pairs (1,2),(3,4),...,(13,14)
    for i in range(7):
        k0 = 1 + 2 * i
        k1 = k0 + 1
        base_r = 4 + 7 * i
        add_strassen_block(base_r, k0, k1)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
