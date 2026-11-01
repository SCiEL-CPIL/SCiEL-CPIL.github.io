"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x15 matrix multiplication.
    Uses Strassen for the first 14 columns (7 blocks of 2) and schoolbook for the last column.
    Total rank: 7*7 + 4 = 53.
    """
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # Strassen formulas for 2x2 x 2x2 multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For each of the 7 Strassen blocks (columns 2b and 2b+1), we add 7 terms.
    # For the last schoolbook block (column 14), we add 4 terms.

    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1

        # M1 = (A00+A11)(B0k0+B1k1)
        U[0][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1

        # M2 = (A10+A11)B0k0
        U[2][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # M3 = A00(B1k0-B1k1)
        U[0][col] = 1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # M4 = A11(B0k0-B0k1)
        U[3][col] = 1
        V[0 * p + k0][col] = 1; V[0 * p + k1][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1

        # M5 = (A00+A01)B1k1
        U[0][col] = 1; U[1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1

        # M6 = (A10-A00)(B0k0+B0k1)
        U[2][col] = 1; U[0][col] = -1
        V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M7 = (A01-A11)(B1k0+B1k1)
        U[1][col] = 1; U[3][col] = -1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    # Last column: schoolbook for k=14
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 14][col] = 1
            W[14 * n + i][col] = 1
            col += 1

    # Convert to numpy arrays
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
