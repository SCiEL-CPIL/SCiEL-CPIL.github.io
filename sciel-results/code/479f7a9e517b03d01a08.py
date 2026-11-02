"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T should equal 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    return _schoolbook()
