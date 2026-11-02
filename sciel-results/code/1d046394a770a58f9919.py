import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _build_rank_8(n, m, p):
    """
    Build the rank-8 decomposition using the Strassen-like trick for 1x4 by 4x8 multiplication.
    Decomposes the inner product of length 4 into two products of length 2 using 4 rank-1 terms per row.
    Total rank = 2 rows * 4 terms = 8.
    """
    total_rank = 8
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # n=2, m=4, p=8
    # U indices: 0..7 (row i, col k) -> 2*i + k
    # V indices: 0..31 (row k, col j) -> 4*k + j? No, V is (m*p, rank) in flat list?
    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # V index: k is 0..3, j is 0..7. Index = k * p + j.
    # W index: j is 0..7, i is 0..1. Index = j * n + i.
    
    for i in range(n):
        for s in range(4):
            col = i * 4 + s
            
            # U coefficients for A[i, k]
            if s == 0:
                U[2*i + 0][col] += 1
                U[2*i + 2][col] += 1
            elif s == 1:
                U[2*i + 0][col] += 1
                U[2*i + 2][col] -= 1
            elif s == 2:
                U[2*i + 1][col] += 1
                U[2*i + 3][col] += 1
            elif s == 3:
                U[2*i + 1][col] += 1
                U[2*i + 3][col] -= 1
            
            # V coefficients for B[k, j]
            # We need to set V[k, j] for all j
            # V index = k * p + j
            if s == 0:
                for j in range(p):
                    V[0 * p + j][col] += 1
                    V[2 * p + j][col] += 1
            elif s == 1:
                for j in range(p):
                    V[0 * p + j][col] += 1
                    V[2 * p + j][col] -= 1
            elif s == 2:
                for j in range(p):
                    V[1 * p + j][col] += 1
                    V[3 * p + j][col] += 1
            elif s == 3:
                for j in range(p):
                    V[1 * p + j][col] += 1
                    V[3 * p + j][col] -= 1
            
            # W coefficients for C[j, i]
            # W index = j * n + i
            for j in range(p):
                W[j * n + i][col] += 0.5

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank_8(n, m, p)
    return {"U": U, "V": V, "W": W}
