"""Matrix multiplication 3x4x4 with Strassen on a 2x2x2 sub-block."""
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4

    # Total rank: 40 schoolbook terms + 7 Strassen terms = 47
    rank = 47

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Schoolbook terms for all (i,j,k) EXCEPT the 2x2x2 block i in {0,1}, j in {0,1}, k in {0,1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Now col should be 40. Add 7 Strassen terms for the 2x2x2 block.
    # Strassen for C = AB where A is 2x2, B is 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A00+A11)(B10-B11)
    # P3 = (A00-A10)B00
    # P4 = A11(B11-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A11-A01)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P6

    # For each Strassen term, we need to set entries in U, V, W.
    # A00 -> index i=0,j=0 -> row 0 in U
    # A01 -> index i=0,j=1 -> row 1 in U
    # A10 -> index i=1,j=0 -> row 4 in U
    # A11 -> index i=1,j=1 -> row 5 in U
    # B00 -> index j=0,k=0 -> row 0 in V
    # B01 -> index j=0,k=1 -> row 1 in V
    # B10 -> index j=1,k=0 -> row 4 in V
    # B11 -> index j=1,k=1 -> row 5 in V
    # C00 -> index k=0,i=0 -> row 0 in W
    # C01 -> index k=1,i=0 -> row 3 in W
    # C10 -> index k=0,i=1 -> row 1 in W
    # C11 -> index k=1,i=1 -> row 4 in W

    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # Each is a list of (row, coeff)
        # P1: (A00+A11)(B00+B11)
        ([(0, 1), (5, 1)], [(0, 1), (5, 1)], [(0, 1), (4, 1)]),
        # P2: (A00+A11)(B10-B11)
        ([(0, 1), (5, 1)], [(4, 1), (5, -1)], [(1, 1), (4, -1)]),
        # P3: (A00-A10)B00
        ([(0, 1), (4, -1)], [(0, 1)], [(3, 1)]),
        # P4: A11(B11-B00)
        ([(5, 1)], [(5, 1), (0, -1)], [(0, 1), (1, 1)]),
        # P5: (A00+A01)B11
        ([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (3, 1)]),
        # P6: (A10-A00)(B00+B01)
        ([(4, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)]),
        # P7: (A11-A01)(B10+B11)
        ([(5, 1), (1, -1)], [(4, 1), (5, 1)], [(0, 1)]),
    ]

    for U_entries, V_entries, W_entries in strassen_terms:
        for row, coeff in U_entries:
            U[row][col] = coeff
        for row, coeff in V_entries:
            V[row][col] = coeff
        for row, coeff in W_entries:
            W[row][col] = coeff
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
