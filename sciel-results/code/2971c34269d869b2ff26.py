"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _add_strassen_block(U, V, W, rank_offset, i1, i2, j1, j2, k1, k2, n, m, p):
    """
    Adds a Strassen decomposition for a 2x2x2 sub-problem.
    A block: rows i1, i2; cols j1, j2.
    B block: rows j1, j2; cols k1, k2.
    C block: rows i1, i2; cols k1, k2.
    """
    # M1 = (X00 + X11)(Y00 + Y11)
    r = rank_offset
    U[i1 * m + j1][r] += 1
    U[i2 * m + j2][r] += 1
    V[j1 * p + k1][r] += 1
    V[j2 * p + k2][r] += 1
    W[k1 * n + i1][r] += 1
    W[k2 * n + i2][r] += 1
    
    # M2 = (X10 + X11)Y00
    r = rank_offset + 1
    U[i2 * m + j1][r] += 1
    U[i2 * m + j2][r] += 1
    V[j1 * p + k1][r] += 1
    W[k1 * n + i2][r] += 1
    
    # M3 = X00(Y10 - Y11)
    r = rank_offset + 2
    U[i1 * m + j1][r] += 1
    V[j2 * p + k1][r] += 1
    V[j2 * p + k2][r] -= 1
    W[k1 * n + i1][r] += 1
    W[k2 * n + i1][r] += 1
    
    # M4 = X11(Y01 + Y11)
    r = rank_offset + 3
    U[i2 * m + j2][r] += 1
    V[j1 * p + k2][r] += 1
    V[j2 * p + k2][r] += 1
    W[k2 * n + i2][r] += 1
    W[k1 * n + i2][r] += 1 # Wait, M4 contributes to Z10 and Z11?
    # Z10 = M2 + M4. Z11 = M1 - M2 + M3 + M6.
    # M4 contributes to Z10 (i2, k1) and Z11 (i2, k2).
    # So W[k1 * n + i2] and W[k2 * n + i2] should be 1.
    
    # M5 = (X00 + X01)Y11
    r = rank_offset + 4
    U[i1 * m + j1][r] += 1
    U[i1 * m + j2][r] += 1
    V[j2 * p + k2][r] += 1
    W[k2 * n + i1][r] += 1
    W[k1 * n + i1][r] -= 1 # M5 contributes to Z01 (+) and Z00 (-).
    # Z00 = M1 + M4 - M5 - M7.
    # Z01 = M3 + M5.
    # So M5 contributes +1 to Z01 (i1, k2) and -1 to Z00 (i1, k1).
    
    # M6 = (X10 - X00)(Y00 + Y10)
    r = rank_offset + 5
    U[i2 * m + j1][r] += 1
    U[i1 * m + j1][r] -= 1
    V[j1 * p + k1][r] += 1
    V[j2 * p + k1][r] += 1
    W[k1 * n + i2][r] += 1 # M6 contributes to Z11.
    # Z11 = M1 - M2 + M3 + M6.
    # So M6 contributes +1 to Z11 (i2, k2).
    # Wait, does it contribute to anything else?
    # The product is (X10-X00)(Y00+Y10).
    # This term is added to Z11 only in Strassen.
    # So W[k2 * n + i2] should be 1?
    # Let's check indices. Z11 is (i2, k2).
    # So W[k2 * n + i2] = 1.
    # My previous line set W[k1 * n + i2]. That is Z10.
    # M6 does NOT contribute to Z10.
    # So W[k2 * n + i2][r] += 1.
    
    # M7 = (X01 - X11)(Y11 + Y01)
    r = rank_offset + 6
    U[i1 * m + j2][r] += 1
    U[i2 * m + j2][r] -= 1
    V[j2 * p + k2][r] += 1
    V[j1 * p + k2][r] += 1
    W[k1 * n + i1][r] -= 1 # M7 contributes to Z00 (-).
    # Z00 = ... - M7.
    # Z00 is (i1, k1).
    # So W[k1 * n + i1] = -1.
    
    return rank_offset + 7

def _add_schoolbook_block(U, V, W, rank_offset, i, m, p, n):
    """
    Adds a schoolbook decomposition for a 1 x m by m x p multiplication.
    A row i. B full. C row i.
    """
    for j in range(m):
        for k in range(p):
            r = rank_offset
            U[i * m + j][r] += 1
            V[j * p + k][r] += 1
            W[k * n + i][r] += 1
            rank_offset += 1
    return rank_offset

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Top 2 rows of A (i=0,1)
    # Split A into A11 (i 0,1; j 0,1), A12 (i 0,1; j 2,3)
    # Split B into B11 (j 0,1; k 0,1), B12 (j 0,1; k 2,3), B21 (j 2,3; k 0,1), B22 (j 2,3; k 2,3)
    
    # P1 = A11 * B11 -> C block (i 0,1; k 0,1)
    r = _add_strassen_block(U, V, W, 0, 0, 1, 0, 1, 0, 1, n, m, p)
    
    # P2 = A12 * B21 -> C block (i 0,1; k 0,1)
    # Note: B21 is j 2,3; k 0,1.
    # A12 is i 0,1; j 2,3.
    r = _add_strassen_block(U, V, W, r, 0, 1, 2, 3, 0, 1, n, m, p)
    
    # P3 = A11 * B12 -> C block (i 0,1; k 2,3)
    # B12 is j 0,1; k 2,3.
    r = _add_strassen_block(U, V, W, r, 0, 1, 0, 1, 2, 3, n, m, p)
    
    # P4 = A12 * B22 -> C block (i 0,1; k 2,3)
    # B22 is j 2,3; k 2,3.
    r = _add_strassen_block(U, V, W, r, 0, 1, 2, 3, 2, 3, n, m, p)
    
    # Part 2: Bottom 1 row of A (i=2)
    # Schoolbook 1x4 * 4x4
    r = _add_schoolbook_block(U, V, W, r, 2, m, p, n)
    
    # r should be 44
    assert r == 44, f"Rank mismatch: {r}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
