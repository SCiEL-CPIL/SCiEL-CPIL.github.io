"""
Strassen-Winograd structured decomposition for 2x4x5 matrix multiplication.

Strategy:
- Use the rank-7 Strassen decomposition for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}).
- For the remaining dimensions (j in {2,3} and k in {2,3,4}), use direct rank-1 terms.
- Total rank = 7 + 2*5 + 2*3 = 7 + 10 + 6 = 23.

The tensor equation is:
  C[i,k] = sum_j A[i,j] * B[j,k]
  => T[i,j,k] = 1 for all i,j,k

Decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
where U_r[i*m+j] = u_{r,i,j}, V_r[j*p+k] = v_{r,j,k}, W_r[k*n+i] = w_{r,k,i}

Strassen for 2x2 block (i in {0,1}, j in {0,1}, k in {0,1}):
  M1 = (A00+A11)(B00+B11)
  M2 = (A10+A11)B00
  M3 = A00(B10-B11)
  M4 = A11(B01-B00)
  M5 = (A00+A01)B11
  M6 = (A10-A00)(B00+B01)
  M7 = (A01-A11)(B10+B11)

  C00 = M1 - M4 + M6 - M7
  C01 = M4 + M5
  C10 = M2 + M3
  C11 = M1 - M3 + M5 - M6

For each Strassen term r, we define:
  u_r[i,j] for i in {0,1}, j in {0,1} (coefficients on A)
  v_r[j,k] for j in {0,1}, k in {0,1} (coefficients on B)
  w_r[k,i] for k in {0,1}, i in {0,1} (coefficients on C)

Then for j in {2,3} or k in {2,3,4}, we use direct terms.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    # Deadline guard
    if time.time() - _START > _DEADLINE - 5.0:
        return _strassen_decomposition()
    return _strassen_decomposition()

def _strassen_decomposition():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for the 2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # Each term has (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) -> coeff
    # v_coeffs: dict mapping (j,k) -> coeff
    # w_coeffs: dict mapping (k,i) -> coeff
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): 1}
        ),
        # M2 = (A10+A11)B00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): 1, (1,0): 1}
        ),
        # M3 = A00(B10-B11)
        (
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M4 = A11(B01-B00)
        (
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): -1, (0,1): 1, (1,1): -1}
        ),
        # M5 = (A00+A01)B11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,1): 1, (1,1): 1}
        ),
        # M6 = (A10-A00)(B00+B01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(0,0): 1, (0,1): -1, (1,0): -1}
        ),
        # M7 = (A01-A11)(B10+B11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): -1, (1,0): 1, (1,1): -1}
        ),
    ]
    
    col = 0
    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
        for (i, j), coeff in u_coeffs.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_coeffs.items():
            V[j * p + k][col] += coeff
        for (k, i), coeff in w_coeffs.items():
            W[k * n + i][col] += coeff
        col += 1
    
    # Direct terms for j in {2,3} and all k in {0,1,2,3,4}
    for j in [2, 3]:
        for k in range(p):
            U[0 * m + j][col] = 1  # i=0
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1  # i=0
            col += 1
            U[1 * m + j][col] = 1  # i=1
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1  # i=1
            col += 1
    
    # Direct terms for j in {0,1} and k in {2,3,4}
    for j in [0, 1]:
        for k in [2, 3, 4]:
            U[0 * m + j][col] = 1  # i=0
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1  # i=0
            col += 1
            U[1 * m + j][col] = 1  # i=1
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1  # i=1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

if __name__ == "__main__":
    result = _strassen_decomposition()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    valid = _verify(result["U"], result["V"], result["W"], n, m, p, rank)
    print(f"Valid: {valid}")
    if not valid:
        # Debug: find which entries fail
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += result["U"][i*m+j][r] * result["V"][j*p+k][r] * result["W"][k*n+i][r]
                    if s != 1:
                        print(f"  T[{i},{j},{k}] = {s} (expected 1)")
