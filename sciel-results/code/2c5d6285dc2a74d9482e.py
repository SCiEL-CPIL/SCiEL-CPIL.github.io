"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Block-partition into six 2x2 @ 2x2 multiplications, each computed
via Strassen's algorithm (rank 7). Total rank = 6 * 7 = 42.

Uses exact integer arithmetic throughout to avoid floating-point issues.
"""
import time

DIMENSIONS = [2, 4, 6]


def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7).
    U1 row: i*2+j, V1 row: j*2+k, W1 row: k*2+i (trace convention).
    All entries are integers in {-1, 0, 1}."""
    U1 = [[0] * 7 for _ in range(4)]
    V1 = [[0] * 7 for _ in range(4)]
    W1 = [[0] * 7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1  # C00 += M1
    W1[3][0] = 1  # C11 += M1

    # M2 = (A10+A11)B00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1  # C10 += M2
    W1[3][1] = -1 # C11 -= M2

    # M3 = A00(B01-B11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1  # C01 += M3
    W1[3][2] = 1  # C11 += M3

    # M4 = A11(B10-B00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1  # C00 += M4
    W1[1][3] = 1  # C10 += M4

    # M5 = (A00+A01)B11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1 # C00 -= M5
    W1[2][4] = 1  # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1  # C00 += M7

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks. All integer."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = [[0] * total_rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 42)

    # Six blocks: for j_block in 0..2 (which 4x2 block of B),
    #             for side in 0..1 (AL or AR, i.e., which 2x2 of A)
    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            # U contribution: A[i, a] where a = side * 2 + j_local
            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local][s]
                        if coeff != 0:
                            U[i * m + a][col_offset + s] += coeff

            # V contribution: B[a, bcol] where a = side * 2 + j_local, bcol = j_block * 2 + k_local
            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local][s]
                        if coeff != 0:
                            V[a * p + bcol][col_offset + s] += coeff

            # W contribution: C[i, bcol] where i in 0..1, bcol = j_block * 2 + k_local
            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i][s]
                        if coeff != 0:
                            W[bcol * n + i][col_offset + s] += coeff

    return U, V, W


def _verify_exact(U, V, W, n, m, p):
    """Verify with exact integer arithmetic: sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for s in range(rank):
                    val += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
                if val != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    U, V, W = _build_rank42()

    # Verify correctness with exact integer arithmetic
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 48 (should never be reached)
    schoolbook_rank = n * m * p
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj][col] = 1
                V_fb[jj * p + kk][col] = 1
                W_fb[kk * n + ii][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
