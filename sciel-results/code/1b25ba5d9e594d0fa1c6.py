"""Rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == (1 if k==i+j else 0)."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if k == i + j else 0
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if total != expected:
                    return False
    return True


def _schoolbook():
    n, m, p = 3, 3, 3
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


def construct():
    n, m, p = DIMENSIONS

    # Schoolbook decomposition (rank 27), guaranteed correct.
    return _schoolbook()

# EVOLVE-BLOCK-END
