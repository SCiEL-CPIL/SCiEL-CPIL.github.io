"""Tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 4, 5]


def construct():
    n, m, p = DIMENSIONS
    N = n * m * p  # 40

    # Build the multiplication tensor T[i, j, k] = 1 if i*m + j == j*p + k
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Kruskal decomposition via rank-revealing SVD
    # We'll use a greedy approach: repeatedly find the best rank-1 component

    best_rank = n * m * p  # upper bound: schoolbook
    best_U = None
    best_V = None
    best_W = None

    # Try to find a decomposition of rank r
    # We'll use a simple approach: build the tensor, then try to decompose

    # Actually, let's use a different strategy:
    # For n=2, we can exploit the structure.
    # The tensor T has a special structure due to the small n.

    # Let's try to find the minimal rank by checking ranks from lower bound to upper bound.
    # Lower bound: max(n*m, m*p, n*p) = max(8, 20, 10) = 20
    # But for matrix multiplication, the lower bound is typically higher.

    # Let's use a practical approach: try to find a decomposition using
    # a combination of Strassen and schoolbook, but optimized for this specific case.

    # For 2x4 * 4x5:
    # We can think of this as:
    # C = A * B where A is 2x4, B is 4x5, C is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]

    # Strategy: split B into B1 (4x2) and B2 (4x3)
    # C1 = A * B1 (2x2), C2 = A * B2 (2x3)
    # C = [C1 | C2]

    # For C1 (2x2): use Strassen with A split into A1 (2x2) and A2 (2x2)
    # C1 = A1*B1_top + A2*B1_bot where B1_top is rows 0,1 and B1_bot is rows 2,3
    # Each 2x2*2x2 uses 7 multiplications (Strassen)
    # Total for C1: 7 + 7 = 14

    # For C2 (2x3): split into C2a (2x1) and C2b (2x2)
    # C2a = A * B_col2 (2x1): schoolbook 2x2*2x1 = 4 mults per half, total 8
    # C2b = A * B_cols3,4 (2x2): same as C1, 14 mults
    # Total for C2: 8 + 14 = 22

    # Grand total: 14 + 22 = 36

    # But can we do better? Let's try a different split.

    # Alternative: split A into A_row0 (1x4) and A_row1 (1x4)
    # C_row0 = A_row0 * B (1x5)
    # C_row1 = A_row1 * B (1x5)
    # Each 1x4 * 4x5 = 5 mults per row, total 10. But that's just schoolbook.

    # Let's stick with the Strassen-based approach but see if we can optimize.

    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(n * p)]

    col = 0

    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A is 2x2 at rows a_r0..a_r0+1, cols a_c0..a_c0+1
        # B is 2x2 at rows b_r0..b_r0+1, cols b_c0..b_c0+1
        # C is 2x2 at rows c_r0..c_r0+1, cols c_c0..c_c0+1

        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1

        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1

        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1

        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1

        # M2 = (a10+a11)b00
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1

        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1

        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1

        # M5 = (a00+a01)b11
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1

        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1
        U[u00][col] -= 1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1
        U[u11][col] -= 1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1

        v0 = b_r0 * p + b_c0
        v1 = (b_r0 + 1) * p + b_c0

        w0 = c_c0 * n + c_r0
        w1 = c_c0 * n + c_r0 + 1

        U[u00][col] = 1
        V[v0][col] = 1
        W[w0][col] = 1
        col += 1

        U[u01][col] = 1
        V[v1][col] = 1
        W[w0][col] = 1
        col += 1

        U[u10][col] = 1
        V[v0][col] = 1
        W[w1][col] = 1
        col += 1

        U[u11][col] = 1
        V[v1][col] = 1
        W[w1][col] = 1
        col += 1

    # C1 = A * B1 where B1 is cols 0,1
    # A1 = A cols 0,1, A2 = A cols 2,3
    # B1_top = B rows 0,1 cols 0,1, B1_bot = B rows 2,3 cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)

    # C2a = A * B_col2 (col 2 of B)
    # B_col2_top = B rows 0,1 col 2, B_col2_bot = B rows 2,3 col 2
    add_2x1(0, 0, 0, 2, 0, 2)
    add_2x1(0, 2, 2, 2, 0, 2)

    # C2b = A * B_cols3,4
    # B2b_top = B rows 0,1 cols 3,4, B2b_bot = B rows 2,3 cols 3,4
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)

    return {"U": U, "V": V, "W": W}
