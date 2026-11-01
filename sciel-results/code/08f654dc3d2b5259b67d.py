"""Rank-56 Strassen decomposition for 2x4 * 4x8 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Splits into 8 independent 2x2 * 2x2 sub-blocks, each using Strassen's
    7-multiplication algorithm. Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    total_rank = 56

    # Allocate factor matrices
    U = [[0] * total_rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]   # 16 x 56

    # Iterate over 8 sub-blocks: u in {0,1} (A column block), v in {0..3} (B col block)
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)

            # A block (2x2) at columns 2u, 2u+1
            a11 = 0 * m + 2 * u
            a12 = 0 * m + 2 * u + 1
            a21 = 1 * m + 2 * u
            a22 = 1 * m + 2 * u + 1

            # B block (2x2) at rows 2u, 2u+1 and cols 2v, 2v+1
            b11 = (2 * u) * p + 2 * v
            b12 = (2 * u) * p + 2 * v + 1
            b21 = (2 * u + 1) * p + 2 * v
            b22 = (2 * u + 1) * p + 2 * v + 1

            # C block (2x2) at cols 2v, 2v+1, with trace convention W[k*n+i, r]
            c11 = (2 * v) * n + 0
            c12 = (2 * v + 1) * n + 0
            c21 = (2 * v) * n + 1
            c22 = (2 * v + 1) * n + 1

            # Strassen 7 bilinear forms for X(2x2) * Y(2x2) -> C(2x2)
            # M1 = (X11+X22)*(Y11+Y22)  -> C11+=1, C22+=1
            U[a11][base_col + 0] += 1
            U[a22][base_col + 0] += 1
            V[b11][base_col + 0] += 1
            V[b22][base_col + 0] += 1
            W[c11][base_col + 0] += 1
            W[c22][base_col + 0] += 1

            # M2 = (X21+X22)*Y11        -> C21+=1, C22-=1
            U[a21][base_col + 1] += 1
            U[a22][base_col + 1] += 1
            V[b11][base_col + 1] += 1
            W[c21][base_col + 1] += 1
            W[c22][base_col + 1] -= 1

            # M3 = X11*(Y12-Y22)        -> C12+=1, C22+=1
            U[a11][base_col + 2] += 1
            V[b12][base_col + 2] += 1
            V[b22][base_col + 2] -= 1
            W[c12][base_col + 2] += 1
            W[c22][base_col + 2] += 1

            # M4 = X22*(Y21-Y11)        -> C11+=1, C21+=1
            U[a22][base_col + 3] += 1
            V[b21][base_col + 3] += 1
            V[b11][base_col + 3] -= 1
            W[c11][base_col + 3] += 1
            W[c21][base_col + 3] += 1

            # M5 = (X11+X12)*Y22        -> C11-=1, C12+=1
            U[a11][base_col + 4] += 1
            U[a12][base_col + 4] += 1
            V[b22][base_col + 4] += 1
            W[c11][base_col + 4] -= 1
            W[c12][base_col + 4] += 1

            # M6 = (X21-X11)*(Y11+Y12)  -> C22+=1
            U[a21][base_col + 5] += 1
            U[a11][base_col + 5] -= 1
            V[b11][base_col + 5] += 1
            V[b12][base_col + 5] += 1
            W[c22][base_col + 5] += 1

            # M7 = (X12-X22)*(Y21+Y22)  -> C11+=1
            U[a12][base_col + 6] += 1
            U[a22][base_col + 6] -= 1
            V[b21][base_col + 6] += 1
            V[b22][base_col + 6] += 1
            W[c11][base_col + 6] += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
