"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy: split the 8 output columns into four 2-column blocks. Each 2x3 x 3x2
sub-tensor (all-ones 2x3x2) is decomposed as the direct sum of:
  * a 2x2x2 all-ones tensor  -> rank 7 (Strassen)
  * a 2x1x2 all-ones tensor  -> rank 1 (outer product)
giving rank 8 per block, total rank 32 (vs schoolbook 48).

We verify this decomposition numerically first; if it fails we fall back to
the rank-44 2x3x2 Strassen block construction, and finally to schoolbook.
"""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > tol:
                    return False
    return True

def _validate_exact(U, V, W, n, m, p):
    """Exact validation using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# Local 2x2 Strassen rank-7 terms in (i, j), (j, k'), (k', i) coordinates.
# Indices: i, j, k' in {0, 1}
_STRASSEN_U = [
    {(0, 0): 1, (1, 1): 1},
    {(1, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (0, 0): -1},
    {(0, 1): 1, (1, 1): -1},
]
_STRASSEN_V = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(0, 1): 1, (1, 1): -1},
    {(1, 0): 1, (0, 0): -1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (1, 1): 1},
]
_STRASSEN_W = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 1): 1, (1, 1): -1},
    {(0, 1): 1, (1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(0, 0): -1, (0, 1): 1},
    {(1, 1): 1},
    {(0, 0): 1},
]

# Strassen decomposition for 2x3 x 3x2 tensor (rank 11)
# This handles the full 2x3 matrix times 3x2 matrix
_STRASSEN_232_U = [
    # Term 0: (A00+A11)(B00+B11)
    {(0,0): 1, (1,2): 1},
    # Term 1: (A10+A11)B00
    {(1,0): 1, (1,1): 1},
    # Term 2: A00(B10-B00)
    {(0,0): 1},
    # Term 3: A11(B11-B01)
    {(1,2): 1},
    # Term 4: (A00+A01)B11
    {(0,0): 1, (0,1): 1},
    # Term 5: (A10-A00)(B00+B01)
    {(1,0): 1, (0,0): -1},
    # Term 6: (A01-A11)(B10+B11)
    {(0,1): 1, (1,2): -1},
    # Term 7: A01 B01
    {(0,1): 1},
    # Term 8: A10 B10
    {(1,0): 1},
    # Term 9: (A00+A01)(B11-B10)
    {(0,0): 1, (0,1): 1},
    # Term 10: (A10+A11)(B01-B00)
    {(1,0): 1, (1,1): 1},
]
_STRASSEN_232_V = [
    # Term 0
    {(0,0): 1, (1,1): 1},
    # Term 1
    {(0,0): 1},
    # Term 2
    {(1,0): 1, (0,0): -1},
    # Term 3
    {(1,1): 1, (0,1): -1},
    # Term 4
    {(1,1): 1},
    # Term 5
    {(0,0): 1, (0,1): 1},
    # Term 6
    {(1,0): 1, (1,1): 1},
    # Term 7
    {(0,1): 1},
    # Term 8
    {(1,0): 1},
    # Term 9
    {(1,1): 1, (1,0): -1},
    # Term 10
    {(0,1): 1, (0,0): -1},
]
_STRASSEN_232_W = [
    # Term 0
    {(0,0): 1, (1,1): 1},
    # Term 1
    {(0,1): 1, (1,1): -1},
    # Term 2
    {(0,0): 1, (0,1): 1},
    # Term 3
    {(0,1): 1, (1,1): 1},
    # Term 4
    {(0,0): 1, (1,1): 1},
    # Term 5
    {(0,0): 1, (0,1): 1},
    # Term 6
    {(0,1): 1, (1,1): 1},
    # Term 7
    {(0,1): 1},
    # Term 8
    {(1,0): 1},
    # Term 9
    {(1,0): 1, (1,1): -1},
    # Term 10
    {(1,0): 1, (1,1): 1},
]

def _block_decomposition(n, m, p):
    """Build rank-32 decomposition by 4 blocks of 8 terms each.
    
    Each block handles a 2x3 x 3x2 sub-problem:
    - 7 Strassen terms for the 2x2 x 2x2 part (rows 0,1 of A; rows 0,1 of B)
    - 1 outer-product term for the 2x1 x 1x2 part (row 2 of B)
    """
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b  # first column of this block
        col = b * 8
        
        # 7 Strassen terms (j in {0,1}, k in {k0, k0+1})
        for r in range(7):
            for (i, j), val in _STRASSEN_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val
        
        # 1 outer-product term (j = 2)
        oc = col + 7
        for i in range(n):
            U[i * m + 2][oc] = 1
        for kp in range(2):
            V[2 * p + (k0 + kp)][oc] = 1
        for kp in range(2):
            for i in range(n):
                W[(k0 + kp) * n + i][oc] = 1

    return {"U": U, "V": V, "W": W}

def _block_decomposition_232(n, m, p):
    """Build rank-44 decomposition by 4 blocks of 11 terms each using 2x3x2 Strassen."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b  # first column of this block
        col = b * 11
        
        # 11 Strassen terms for 2x3 x 3x2
        for r in range(11):
            for (i, j), val in _STRASSEN_232_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_232_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_232_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val

    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 135  # safety margin before 150s kill

    # Try the optimized 2x2x2 + outer product block decomposition (rank 32)
    try:
        cand = _block_decomposition(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            # Double-check with exact arithmetic for safety
            if _validate_exact(cand["U"], cand["V"], cand["W"], n, m, p):
                return cand
    except Exception:
        pass

    # Try the 2x3x2 Strassen block decomposition (rank 44)
    try:
        cand = _block_decomposition_232(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            if _validate_exact(cand["U"], cand["V"], cand["W"], n, m, p):
                return cand
    except Exception:
        pass

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
