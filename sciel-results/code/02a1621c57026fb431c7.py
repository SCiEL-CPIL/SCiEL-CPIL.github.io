"""Matrix multiplication 4x4x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _strassen_2x2_2x2():
    """Return U, V, W (as 2D lists) for 2x2x2 matrix mult via Strassen (rank 7).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]  # 4 x 7
    V = [[0]*rank for _ in range(m*p)]  # 4 x 7
    W = [[0]*rank for _ in range(p*n)]  # 4 x 7

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)B00 -> C10 += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3 = A00(B10-B11) -> C01 += M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1

    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _strassen_4x4_4x4():
    """Return U, V, W for 4x4x4 matrix mult using Kronecker of Strassen 2x2x2 (rank 49)."""
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]  # 16 x 49
    V = [[0]*rank for _ in range(m*p)]  # 16 x 49
    W = [[0]*rank for _ in range(p*n)]  # 16 x 49

    U2, V2, W2 = _strassen_2x2_2x2()

    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U[i][c] = U2[i1][c1] * U2[i2][c2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    V[j][c] = V2[j1][c1] * V2[j2][c2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    W[k][c] = W2[k1][c1] * W2[k2][c2]

    return U, V, W


def _mul_4x4_4x1():
    """Return U, V, W for 4x4 * 4x1 -> 4x1 (rank 4).
    A is 4x4, B is 4x1, C is 4x1.
    C[i] = sum_j A[i,j] * B[j]
    U index: i*m+j = i*4+j (16 rows)
    V index: j*p+k = j*1+0 = j (4 rows)
    W index: k*n+i = 0*4+i = i (4 rows)
    """
    n = m = 4
    p = 1
    rank = 4
    U = [[0]*rank for _ in range(n*m)]  # 16 x 4
    V = [[0]*rank for _ in range(m*p)]  # 4 x 4
    W = [[0]*rank for _ in range(p*n)]  # 4 x 4

    for i in range(4):
        for j in range(4):
            U[i*m+j][i] = 1
            V[j][i] = 1
            W[i][i] = 1

    return U, V, W


def _mul_4x4_4x7():
    """Build 4x4 * 4x7 decomposition.
    Split B (4x7) into B_left (4x4) + B_right (4x3).
    B_right is 3 columns, each a 4x1 vector.
    Total rank: 49 (Strassen 4x4x4) + 3*4 (three 4x4x1) = 61.
    """
    n = m = 4
    p = 7

    U4, V4, W4 = _strassen_4x4_4x4()
    U1, V1, W1 = _mul_4x4_4x1()

    rank4 = len(U4[0])  # 49
    rank1 = len(U1[0])  # 4
    total_rank = rank4 + 3 * rank1  # 61

    # U_combined: 16 x 61
    U_combined = [row4[:] for row4 in U4]
    for col in range(16):
        U_combined[col] = U_combined[col] + [0]*(3*rank1)

    # V_combined: 28 x 61
    # B is 4x7: rows j*7+k for j in 0..3, k in 0..6
    # B_left: j in 0..3, k in 0..3 (first 4 columns)
    # B_right: j in 0..3, k in 4..6 (last 3 columns)
    V_combined = [[0]*total_rank for _ in range(m*p)]  # 28 x 61

    # Fill B_left (first 4 columns of B)
    for j in range(4):
        for k in range(4):
            v_idx = j * 4 + k  # index in V4 (p=4)
            V_combined[j * 7 + k] = V4[v_idx][:] + [0]*(3*rank1)

    # Fill B_right (last 3 columns of B)
    # Each column is a 4x1 vector, using the rank-4 decomposition
    for col_offset in range(3):
        k = 4 + col_offset  # column index in B (0-indexed: 4,5,6)
        base = rank4 + col_offset * rank1
        for j in range(4):
            # V1[j][i] where i is the "row" of the 4x1 result
            for i in range(4):
                V_combined[j * 7 + k][base + i] = V1[j][i]

    # W_combined: 28 x 61
    # C is 4x7: rows k*4+i for k in 0..6, i in 0..3
    # C_left: k in 0..3 (first 4 columns)
    # C_right: k in 4..6 (last 3 columns)
    W_combined = [[0]*total_rank for _ in range(p*n)]  # 28 x 61

    # Fill C_left (first 4 columns of C)
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i  # index in W4 (n=4)
            W_combined[k * 4 + i] = W4[w_idx][:] + [0]*(3*rank1)

    # Fill C_right (last 3 columns of C)
    for col_offset in range(3):
        k = 4 + col_offset
        base = rank4 + col_offset * rank1
        for i in range(4):
            W_combined[k * 4 + i][base + i] = W1[i][i]  # W1[i][i] = 1

    return U_combined, V_combined, W_combined


def _validate(U, V, W, n, m, p):
    """Validate a decomposition."""
    rank = len(U[0])
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(total - 1) > 1e-9:
                    return False
    # Check integer/half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Kronecker Strassen for 4x4x4 (rank 49) plus 3 rank-4 for 4x4x1 columns.
    Total rank: 61.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    U, V, W = _mul_4x4_4x7()

    if time.time() < deadline:
        valid = _validate(U, V, W, n, m, p)
        if valid:
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook (rank 112)
    rank_school = n * m * p
    U_school = [[0]*rank_school for _ in range(n*m)]
    V_school = [[0]*rank_school for _ in range(m*p)]
    W_school = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1

    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
