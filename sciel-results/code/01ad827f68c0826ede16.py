"""
Rank-minimal exact tensor decomposition for 2x2 matrix multiplication.

The 2x2 matrix multiplication tensor has tensor rank 7 (Strassen).
We return the verified rank-7 Strassen decomposition.

Interface: construct() -> {'U', 'V', 'W'}:
  U: shape (n*m, rank) = (4, 7)
  V: shape (m*p, rank) = (4, 7)
  W: shape (p*n, rank) = (4, 7)

Indexing convention:
  U[i*m + a, r] : row i of A, column a
  V[a*p + k, r] : row a of B, column k
  W[k*n + i, r] : column k of C, row i (official trace convention)

The decomposition satisfies:
  (A*B)[i, k] = sum_{a} sum_{r} U[i*m + a, r] * V[a*p + k, r] * W[k*n + i, r]
  for all i in {0,1}, k in {0,1}
"""

import numpy as np

DIMENSIONS = [2, 2, 2]


def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.

    Strassen's 7 bilinear forms:
      M1 = (A00 + A11)(B00 + B11)
      M2 = (A10 + A11)B00
      M3 = A00(B01 - B11)
      M4 = A11(B10 - B00)
      M5 = (A00 + A01)B11
      M6 = (A10 - A00)(B00 + B01)
      M7 = (A01 - A11)(B10 + B11)

    Output:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7

    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # (4, 7)

    # Indexing:
    # U[i*m + a, r] where i in {0,1}, a in {0,1}
    #   U[0, r] = A00, U[1, r] = A01, U[2, r] = A10, U[3, r] = A11
    # V[a*p + k, r] where a in {0,1}, k in {0,1}
    #   V[0, r] = B00, V[1, r] = B01, V[2, r] = B10, V[3, r] = B11
    # W[k*n + i, r] where k in {0,1}, i in {0,1}
    #   W[0, r] = C00 coeff, W[1, r] = C10 coeff, W[2, r] = C01 coeff, W[3, r] = C11 coeff

    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11

    # M2 = (A10 + A11)B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00

    # M3 = A00(B01 - B11)
    U[0][2] = 1   # A00
    V[1][2] = 1   # B01
    V[3][2] = -1  # -B11

    # M4 = A11(B10 - B00)
    U[3][3] = 1   # A11
    V[2][3] = 1   # B10
    V[0][3] = -1  # -B00

    # M5 = (A00 + A01)B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11

    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # -A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01

    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # -A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11

    # W coefficients: C[i,k] = sum_r W[k*n+i, r] * M_r
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7

    # C[1,0] = M2 + M4
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4

    # C[0,1] = M3 + M5
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5

    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6

    return {"U": U, "V": V, "W": W}
