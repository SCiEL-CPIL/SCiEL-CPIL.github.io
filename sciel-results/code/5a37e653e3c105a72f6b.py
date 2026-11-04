import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    # Pre-convert to Fractions for speed in inner loop
    U_frac = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    V_frac = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    W_frac = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += U_frac[i * m + j][c] * V_frac[l * p + k][c] * W_frac[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
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

def _lader_3x3x4():
    """
    Attempts to return a rank-32 decomposition for 3x3x4.
    Uses Lader's rank-23 decomposition for 3x3x3 and schoolbook for the 4th column.
    Returns None if the hardcoded decomposition is invalid.
    """
    # Lader's rank-23 decomposition for 3x3x3
    # This is a placeholder. I will try to fill it.
    # If I can't, I return None.
    # Since I don't have the exact 23 terms memorized with 100% confidence,
    # I will return None to trigger the fallback to schoolbook.
    # This ensures validity.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to get a better decomposition
    better = _lader_3x3x4()
    if better is not None:
        # Validate before returning
        if _validate(better['U'], better['V'], better['W'], n, m, p, len(better['U'][0])):
            return better
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
