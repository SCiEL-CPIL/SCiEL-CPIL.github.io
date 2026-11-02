"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 6]


def _strassen_222():
    """Return U, V, W for 2x2x2 matrix multiplication via Strassen (rank 7).
    Local indices: A row-major 2x2 (0..3), B row-major 2x2 (0..3), C row-major 2x2 (0..3).
    """
    # Strassen's 7 products:
    # M1 = (A11)(B11+B22)
    # M2 = (A11+A22)(B22)
    # M3 = (A12)(B21-B22)
    # M4 = (A21)(B11-B12)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A12-A22)(B21+B22)
    # M7 = (A11-A21)(B12+B21)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)

    # M1 = (A11)(B11+B22)
    U[0, 0] = 1  # A11
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11

    # M2 = (A11+A22)(B22)
    U[0, 1] = 1  # A11
    U[3, 1] = 1  # A22
    V[3, 1] = 1  # B22
    W[2, 1] = 1  # C21

    # M3 = (A12)(B21-B22)
    U[1, 2] = 1  # A12
    V[2, 2] = 1  # B21
    V[3, 2] = -1  # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22

    # M4 = (A21)(B11-B12)
    U[2, 3] = 1  # A21
    V[0, 3] = 1  # B11
    V[1, 3] = -1  # B12
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21

    # M5 = (A11+A22)(B11+B22)
    U[0, 4] = 1  # A11
    U[3, 4] = 1  # A22
    V[0, 4] = 1  # B11
    V[3, 4] = 1  # B22
    W[0, 4] = -1  # C11
    W[1, 4] = 1  # C12

    # M6 = (A12-A22)(B21+B22)
    U[1, 5] = 1  # A12
    U[3, 5] = -1  # A22
    V[2, 5] = 1  # B21
    V[3, 5] = 1  # B22
    W[3, 5] = 1  # C22

    # M7 = (A11-A21)(B12+B21)
    U[0, 6] = 1  # A11
    U[2, 6] = -1  # A21
    V[1, 6] = 1  # B12
    V[2, 6] = 1  # B21
    W[0, 6] = 1  # C11

    return U, V, W


def _trivial_233():
    """Return U, V, W for 2x3x3 matrix multiplication (trivial, rank 18).
    Local indices: A row-major 2x3 (0..5), B row-major 3x3 (0..8), C row-major 2x3 (0..5).
    """
    U = np.zeros((6, 18), dtype=int)
    V = np.zeros((9, 18), dtype=int)
    W = np.zeros((6, 18), dtype=int)

    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[i * 3 + k, col] = 1
                col += 1

    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6

    # We partition the 5 middle dimension into two parts: 2 and 3.
    # A is 2x5, B is 5x6.
    # Split A into A1 (2x2) and A2 (2x3).
    # Split B into B1 (2x6) and B2 (3x6).
    # C = A1*B1 + A2*B2.
    #
    # For A1*B1: A1 is 2x2, B1 is 2x6. Split B1 into three 2x2 blocks: B11, B12, B13.
    #   Use Strassen for each 2x2x2 block. Rank = 7 per block, total 21.
    #
    # For A2*B2: A2 is 2x3, B2 is 3x6. Split B2 into two 3x3 blocks: B21, B22.
    #   Use trivial 2x3x3 for each. Rank = 18 per block, total 36.
    #
    # Total rank = 21 + 36 = 57.

    # Initialize full matrices
    total_rank = 57
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # --- Part 1: A1 * B1 using Strassen on three 2x2x2 blocks ---
    # A1 is rows 0-1 of A, cols 0-1. Global A index: i*m + j = i*5 + j.
    # B1 is rows 0-1 of B, cols 0-5. Global B index: j*p + k = j*6 + k.
    # C is rows 0-1, cols 0-5. Global C index: k*n + i = k*2 + i.

    sU, sV, sW = _strassen_222()

    # Block 1: B11 (cols 0-1 of B1, i.e., cols 0-1 of B)
    for r in range(7):
        col = r
        for i in range(2):
            for j in range(2):
                U[i * 5 + j, col] += sU[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * 6 + k, col] += sV[j * 2 + k, r]
        for i in range(2):
            for k in range(2):
                W[k * 2 + i, col] += sW[i * 2 + k, r]

    # Block 2: B12 (cols 2-3 of B1, i.e., cols 2-3 of B)
    for r in range(7):
        col = 7 + r
        for i in range(2):
            for j in range(2):
                U[i * 5 + j, col] += sU[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * 6 + (2 + k), col] += sV[j * 2 + k, r]
        for i in range(2):
            for k in range(2):
                W[(2 + k) * 2 + i, col] += sW[i * 2 + k, r]

    # Block 3: B13 (cols 4-5 of B1, i.e., cols 4-5 of B)
    for r in range(7):
        col = 14 + r
        for i in range(2):
            for j in range(2):
                U[i * 5 + j, col] += sU[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * 6 + (4 + k), col] += sV[j * 2 + k, r]
        for i in range(2):
            for k in range(2):
                W[(4 + k) * 2 + i, col] += sW[i * 2 + k, r]

    # --- Part 2: A2 * B2 using trivial 2x3x3 on two blocks ---
    # A2 is rows 0-1 of A, cols 2-4. Global A index: i*5 + (2+j) where j=0,1,2.
    # B2 is rows 2-4 of B, cols 0-5.
    # Block B21: cols 0-2 of B.
    # Block B22: cols 3-5 of B.

    tU, tV, tW = _trivial_233()

    # Block B21 (cols 0-2 of B)
    for r in range(18):
        col = 21 + r
        for i in range(2):
            for j in range(3):
                U[i * 5 + (2 + j), col] += tU[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V[(2 + j) * 6 + k, col] += tV[j * 3 + k, r]
        for i in range(2):
            for k in range(3):
                W[k * 2 + i, col] += tW[i * 3 + k, r]

    # Block B22 (cols 3-5 of B)
    for r in range(18):
        col = 39 + r
        for i in range(2):
            for j in range(3):
                U[i * 5 + (2 + j), col] += tU[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V[(2 + j) * 6 + (3 + k), col] += tV[j * 3 + k, r]
        for i in range(2):
            for k in range(3):
                W[(3 + k) * 2 + i, col] += tW[i * 3 + k, r]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
