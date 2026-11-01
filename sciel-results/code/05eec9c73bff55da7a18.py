import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Rank 56 construction using 8 independent Strassen 2x2x2 blocks
    U, V, W = _build_rank_56(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition.
    U: n*m x rank (8 x 56)
    V: m*p x rank (32 x 56)
    W: p*n x rank (16 x 56)
    """
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # We have 8 blocks.
    # u corresponds to the pair of columns in A (and rows in B)
    # v corresponds to the pair of columns in B (and columns in C)
    # u in {0, 1}, v in {0, 1, 2, 3}
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A indices: Row i, Col j. Index = i*m + j = i*4 + j
            # Block uses A cols 2u, 2u+1
            a11_idx = 0 * m + 2 * u
            a12_idx = 0 * m + 2 * u + 1
            a21_idx = 1 * m + 2 * u
            a22_idx = 1 * m + 2 * u + 1
            
            # B indices: Row j, Col k. Index = j*p + k = j*8 + k
            # Block uses B rows 2u, 2u+1 and cols 2v, 2v+1
            b11_idx = 2 * u * p + 2 * v
            b12_idx = 2 * u * p + 2 * v + 1
            b21_idx = (2 * u + 1) * p + 2 * v
            b22_idx = (2 * u + 1) * p + 2 * v + 1
            
            # C indices: Row i, Col k. Index = k*n + i
            # Block uses C cols 2v, 2v+1 and rows 0, 1
            c11_idx = 2 * v * n + 0
            c12_idx = (2 * v + 1) * n + 0
            c21_idx = 2 * v * n + 1
            c22_idx = (2 * v + 1) * n + 1
            
            # Strassen 7 multiplications for 2x2 matrix mult
            # M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
    
    return U, V, W
