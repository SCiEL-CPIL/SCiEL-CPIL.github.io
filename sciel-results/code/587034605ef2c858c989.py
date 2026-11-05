"""Block-decomposed matrix multiplication 3x5x5 using Strassen for 2x2 sub-blocks."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130

    n, m, p = DIMENSIONS
    # Total rank: Strassen(2,2,2)=7 + schoolbook(2,3,2)=12 + schoolbook(1,2,2)=4 + schoolbook(1,3,2)=6
    rank = 7 + 12 + 4 + 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Strassen 2x2x2: rows 0-1, inner 0-1, output 0-1
    # m1 = (a+d)(e+h)
    U[0*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*2+0][col] = 1; V[1*2+1][col] = 1
    W[0*2+0][col] += 1; W[1*2+1][col] += 1
    col += 1
    # m2 = (b+c)d
    U[0*2+1][col] = 1; U[1*2+0][col] = 1
    V[1*2+1][col] = 1
    W[0*2+1][col] += 1
    col += 1
    # m3 = a(f-h)
    U[0*2+0][col] = 1
    V[0*2+1][col] = 1; V[1*2+1][col] = -1
    W[1*2+0][col] += 1
    col += 1
    # m4 = d(g-e)
    U[1*2+1][col] = 1
    V[1*2+0][col] = 1; V[0*2+0][col] = -1
    W[0*2+1][col] += 1
    col += 1
    # m5 = (a+b)h
    U[0*2+0][col] = 1; U[0*2+1][col] = 1
    V[1*2+1][col] = 1
    W[1*2+0][col] += 1
    col += 1
    # m6 = (c-d)(g+h)
    U[1*2+0][col] = 1; U[1*2+1][col] = -1
    V[1*2+0][col] = 1; V[1*2+1][col] = 1
    W[1*2+1][col] += 1
    col += 1
    # m7 = (b-d)(e+f)
    U[0*2+1][col] = 1; U[1*2+1][col] = -1
    V[0*2+0][col] = 1; V[0*2+1][col] = 1
    W[0*2+0][col] += 1; W[0*2+1][col] += -1
    col += 1

    # Schoolbook 2x3x2: rows 0-1, inner 2-4, output 0-1
    for ii in range(2):
        for jj in range(3):
            for kk in range(2):
                aj = 2 + jj
                U[ii * m + aj][col] = 1
                V[aj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1

    # Schoolbook 1x2x2: row 2, inner 0-1, output 0-1
    for jj in range(2):
        for kk in range(2):
            U[2 * m + jj][col] = 1
            V[jj * p + kk][col] = 1
            W[kk * n + 2][col] = 1
            col += 1

    # Schoolbook 1x3x2: row 2, inner 2-4, output 0-1
    for jj in range(3):
        for kk in range(2):
            aj = 2 + jj
            U[2 * m + aj][col] = 1
            V[aj * p + kk][col] = 1
            W[kk * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
