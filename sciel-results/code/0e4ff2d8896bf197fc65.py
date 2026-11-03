"""Tensor decomposition for 3x3 * 3x8 matrix multiplication.

Design family: schoolbook decomposition (rank 72) with 4 non-overlapping
2x2*2x2 Strassen blocks embedded. Each block replaces 8 schoolbook rank-1
terms with 7 exact rank-1 terms, saving 1 rank per block. Total rank: 68.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    rank = n * m * p  # 72

    # Build schoolbook first.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col_of = {}
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col_of[(i, j, k)] = column
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Define 4 non-overlapping 2x2*2x2 Strassen blocks.
    # Each block: A rows {i1,i2}, A cols {j1,j2}, B rows {j1,j2}, B cols {k1,k2}
    blocks = [
        (0, 1, 0, 1, 0, 1),  # i1=0,i2=1, j1=0,j2=1, k1=0,k2=1
        (0, 1, 0, 1, 2, 3),  # k1=2,k2=3
        (0, 1, 0, 1, 4, 5),  # k1=4,k2=5
        (0, 1, 0, 1, 6, 7),  # k1=6,k2=7
    ]

    # Collect all (i,j,k) tuples covered by the blocks.
    covered = set()
    for (i1, i2, j1, j2, k1, k2) in blocks:
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Remove covered schoolbook columns, keep the rest.
    U2 = []
    V2 = []
    W2 = []
    for c in range(rank):
        # Find which (i,j,k) this column corresponds to.
        # Schoolbook column c has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
        # We can identify it by checking which (i,j,k) has this column.
        # Simpler: build a reverse map.
        pass

    # Build reverse map: column -> (i,j,k)
    col_to_ijk = {}
    for (i, j, k), c in col_of.items():
        col_to_ijk[c] = (i, j, k)

    for c in range(rank):
        if col_to_ijk[c] in covered:
            continue
        U2.append([U[r][c] for r in range(n * m)])
        V2.append([V[r][c] for r in range(m * p)])
        W2.append([W[r][c] for r in range(p * n)])

    # Now add 7 Strassen columns per block.
    for (i1, i2, j1, j2, k1, k2) in blocks:
        # A[i1,j1] = U row i1*m+j1, A[i1,j2] = U row i1*m+j2
        # A[i2,j1] = U row i2*m+j1, A[i2,j2] = U row i2*m+j2
        # B[j1,k1] = V row j1*p+k1, B[j1,k2] = V row j1*p+k2
        # B[j2,k1] = V row j2*p+k1, B[j2,k2] = V row j2*p+k2
        # W for output: C[i1,k1] += ..., W row k1*n+i1, etc.

        def Ucol(a11, a12, a21, a22):
            col = [0] * (n * m)
            col[i1 * m + j1] = a11
            col[i1 * m + j2] = a12
            col[i2 * m + j1] = a21
            col[i2 * m + j2] = a22
            return col

        def Vcol(b11, b12, b21, b22):
            col = [0] * (m * p)
            col[j1 * p + k1] = b11
            col[j1 * p + k2] = b12
            col[j2 * p + k1] = b21
            col[j2 * p + k2] = b22
            return col

        def Wcol(c11, c12, c21, c22):
            col = [0] * (p * n)
            col[k1 * n + i1] = c11
            col[k1 * n + i2] = c12
            col[k2 * n + i1] = c21
            col[k2 * n + i2] = c22
            return col

        # M1 = (A[i1,j1] + A[i2,j2]) (B[j1,k1] + B[j2,k2])
        U2.append(Ucol(1, 0, 0, 1))
        V2.append(Vcol(1, 0, 0, 1))
        W2.append(Wcol(1, 0, 0, 1))

        # M2 = (A[i2,j1] + A[i2,j2]) B[j1,k1]
        U2.append(Ucol(0, 0, 1, 1))
        V2.append(Vcol(1, 0, 0, 0))
        W2.append(Wcol(1, 0, 0, 0))

        # M3 = A[i1,j1] (B[j2,k1] - B[j2,k2])
        U2.append(Ucol(1, 0, 0, 0))
        V2.append(Vcol(0, 0, 1, -1))
        W2.append(Wcol(1, 0, 0, 0))

        # M4 = A[i2,j2] (B[j1,k2] - B[j1,k1])
        U2.append(Ucol(0, 0, 0, 1))
        V2.append(Vcol(0, 1, 0, -1))
        W2.append(Wcol(0, 0, 1, 0))

        # M5 = (A[i1,j1] + A[i1,j2]) B[j2,k2]
        U2.append(Ucol(1, 1, 0, 0))
        V2.append(Vcol(0, 0, 0, 1))
        W2.append(Wcol(0, 1, 0, 0))

        # M6 = (A[i2,j1] - A[i1,j1]) (B[j1,k1] + B[j1,k2])
        U2.append(Ucol(-1, 0, 1, 0))
        V2.append(Vcol(1, 1, 0, 0))
        W2.append(Wcol(1, 0, 0, 0))

        # M7 = (A[i1,j2] - A[i2,j2]) (B[j2,k1] + B[j2,k2])
        U2.append(Ucol(0, 1, 0, -1))
        V2.append(Vcol(0, 0, 1, 1))
        W2.append(Wcol(0, 0, 1, 0))

    final_rank = len(U2)

    # Verify: for all (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(final_rank):
                        s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None

    ok, info = verify()
    if not ok:
        return {"U": U, "V": V, "W": W}

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U2, "V": V2, "W": W2}
# EVOLVE-BLOCK-END
