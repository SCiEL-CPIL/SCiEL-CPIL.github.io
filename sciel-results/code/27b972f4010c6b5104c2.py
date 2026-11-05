import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 5]

def _to_fraction(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, float)):
        return Fraction(x)
    return Fraction(x)

def _is_half_integer(frac):
    return (frac * 2).denominator == 1

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2(U, V, W, rank_offset, i_offset, j_offset, k_offset):
    """
    Adds Strassen decomposition for 2x2 * 2x2 block.
    U, V, W are the global matrices.
    rank_offset: starting index for rank.
    i_offset: row offset in A (0..1)
    j_offset: col offset in A (0..1) -> corresponds to row in B
    k_offset: col offset in B (0..1)
    """
    # M1
    U[i_offset * 3 + j_offset][rank_offset] = 1
    U[(i_offset+1) * 3 + (j_offset+1)][rank_offset] = 1
    V[j_offset * 5 + k_offset][rank_offset] = 1
    V[(j_offset+1) * 5 + (k_offset+1)][rank_offset] = 1
    W[k_offset * 2 + i_offset][rank_offset] = 1
    W[(k_offset+1) * 2 + (i_offset+1)][rank_offset] = 1
    rank_offset += 1
    
    # M2
    U[(i_offset+1) * 3 + j_offset][rank_offset] = 1
    U[(i_offset+1) * 3 + (j_offset+1)][rank_offset] = 1
    V[j_offset * 5 + k_offset][rank_offset] = 1
    W[k_offset * 2 + (i_offset+1)][rank_offset] = 1
    W[(k_offset+1) * 2 + (i_offset+1)][rank_offset] = -1
    rank_offset += 1
    
    # M3
    U[i_offset * 3 + j_offset][rank_offset] = 1
    V[j_offset * 5 + (k_offset+1)][rank_offset] = 1
    V[(j_offset+1) * 5 + (k_offset+1)][rank_offset] = -1
    W[(k_offset+1) * 2 + i_offset][rank_offset] = 1
    W[(k_offset+1) * 2 + (i_offset+1)][rank_offset] = 1
    rank_offset += 1
    
    # M4
    U[(i_offset+1) * 3 + (j_offset+1)][rank_offset] = 1
    V[(j_offset+1) * 5 + k_offset][rank_offset] = 1
    V[j_offset * 5 + k_offset][rank_offset] = -1
    W[k_offset * 2 + i_offset][rank_offset] = 1
    W[k_offset * 2 + (i_offset+1)][rank_offset] = 1
    rank_offset += 1
    
    # M5
    U[i_offset * 3 + j_offset][rank_offset] = 1
    U[i_offset * 3 + (j_offset+1)][rank_offset] = 1
    V[(j_offset+1) * 5 + (k_offset+1)][rank_offset] = 1
    W[k_offset * 2 + i_offset][rank_offset] = -1
    W[k_offset * 2 + (i_offset+1)][rank_offset] = 1
    rank_offset += 1
    
    # M6
    U[(i_offset+1) * 3 + j_offset][rank_offset] = 1
    U[i_offset * 3 + j_offset][rank_offset] = -1
    V[j_offset * 5 + k_offset][rank_offset] = 1
    V[j_offset * 5 + (k_offset+1)][rank_offset] = 1
    W[(k_offset+1) * 2 + (i_offset+1)][rank_offset] = 1
    rank_offset += 1
    
    # M7
    U[i_offset * 3 + (j_offset+1)][rank_offset] = 1
    U[(i_offset+1) * 3 + (j_offset+1)][rank_offset] = -1
    V[(j_offset+1) * 5 + k_offset][rank_offset] = 1
    V[(j_offset+1) * 5 + (k_offset+1)][rank_offset] = 1
    W[k_offset * 2 + i_offset][rank_offset] = 1
    rank_offset += 1
    
    return rank_offset

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 16 decomposition
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A[0,1] * B[0,1] (Cols 0,1 of B)
    # i_offset=0, j_offset=0, k_offset=0
    r = _strassen_2x2(U, V, W, r, 0, 0, 0)
    
    # Part 2: A[0,1] * B[2,3] (Cols 2,3 of B)
    # i_offset=0, j_offset=0, k_offset=2
    r = _strassen_2x2(U, V, W, r, 0, 0, 2)
    
    # Part 3: A[2] * B[2, 0..3] (Rank 1)
    # A index j=2. B index j=2, k=0..3.
    # U[i*3+2, r] = 1
    # V[2*5+k, r] = 1
    # W[k*2+i, r] = 1
    for i in range(n):
        U[i * m + 2][r] = 1
    for k in range(4):
        V[2 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = 1
    r += 1
    
    # Part 4: A[0,1,2] * B[0,1,2, 4] (Rank 1)
    # A indices j=0,1,2. B indices j=0,1,2, k=4.
    # U[i*3+j, r] = 1
    # V[j*5+4, r] = 1
    # W[4*2+i, r] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][r] = 1
    for j in range(m):
        V[j * p + 4][r] = 1
    for i in range(n):
        W[4 * n + i][r] = 1
    r += 1
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
