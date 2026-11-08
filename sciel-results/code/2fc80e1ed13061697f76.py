import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    U_t = []
    V_t = []
    W_t = []

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)

        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    # For each column block of B (3 columns each for first 2 blocks, 2 for last)
    # We decompose 3x3 * 3xk into:
    #   Strassen for A[0:2,0:2] * B[0:2, c:c+2]  (if k >= 2)
    #   Direct for remaining terms
    #
    # For a 3x3 * 3x3 block (c=0, c=3):
    #   Strassen for A[0:2,0:2] * B[0:2, c:c+2] -> 7 terms
    #   Direct: A[0:2,2] * B[2, c:c+3] -> 2*3 = 6 terms
    #   Direct: A[2,0:2] * B[0:2, c:c+3] -> 2*3 = 6 terms
    #   Direct: A[2,2] * B[2, c:c+3] -> 3 terms
    #   Total: 7 + 6 + 6 + 3 = 22
    #
    # For a 3x3 * 3x2 block (c=6):
    #   Strassen for A[0:2,0:2] * B[0:2, 6:8] -> 7 terms
    #   Direct: A[0:2,2] * B[2, 6:8] -> 2*2 = 4 terms
    #   Direct: A[2,0:2] * B[0:2, 6:8] -> 2*2 = 4 terms
    #   Direct: A[2,2] * B[2, 6:8] -> 2 terms
    #   Total: 7 + 4 + 4 + 2 = 17
    #
    # Grand total: 22 + 22 + 17 = 61

    # Block 1: columns 0,1,2
    strassen_decomp(0, 0, 0, 0)
    for i in range(2):
        for k in range(3):
            direct_term(i, 2, k)
    for j in range(2):
        for k in range(3):
            direct_term(2, j, k)
    for k in range(3):
        direct_term(2, 2, k)

    # Block 2: columns 3,4,5
    strassen_decomp(0, 0, 0, 3)
    for i in range(2):
        for k in range(3, 6):
            direct_term(i, 2, k)
    for j in range(2):
        for k in range(3, 6):
            direct_term(2, j, k)
    for k in range(3, 6):
        direct_term(2, 2, k)

    # Block 3: columns 6,7
    strassen_decomp(0, 0, 0, 6)
    for i in range(2):
        for k in range(6, 8):
            direct_term(i, 2, k)
    for j in range(2):
        for k in range(6, 8):
            direct_term(2, j, k)
    for k in range(6, 8):
        direct_term(2, 2, k)

    rank = len(W_t)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v * 2) / 2) < 1e-9:
                    nv = round(v * 2) / 2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
