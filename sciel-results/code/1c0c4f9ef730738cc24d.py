import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Construct rank-56 decomposition for 2x4 * 4x8 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # 2x4 = 2 blocks of size 2x2 along the middle dimension
    # 4x8 = 2 blocks of size 2x2 along the middle dim, 4 blocks of size 2x2 along the last dim
    # So we have 2 * 4 = 8 blocks of 2x2 * 2x2 * 2x2
    # Each 2x2*2x2 has rank 7 (Strassen)
    # Total rank = 8 * 7 = 56
    
    total_rank = 56
    
    # U: shape (n*m, rank) = (8, 56)
    # V: shape (m*p, rank) = (32, 56)
    # W: shape (p*n, rank) = (16, 56)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Index conventions:
    # A is 2x4: A[i,j] where i in 0..1, j in 0..3
    #   U index for A[i,j]: i*m + j = i*4 + j
    # B is 4x8: B[j,k] where j in 0..3, k in 0..7
    #   V index for B[j,k]: j*p + k = j*8 + k
    # C is 2x8: C[i,k] where i in 0..1, k in 0..7
    #   W index for C[i,k]: k*n + i = k*2 + i (trace convention)
    
    # Block structure:
    # u indexes the 2x2 block along the middle dimension (j dimension)
    #   u=0: j in {0,1}, u=1: j in {2,3}
    # v indexes the 2x2 block along the last dimension (k dimension)
    #   v=0: k in {0,1}, v=1: k in {2,3}, v=2: k in {4,5}, v=3: k in {6,7}
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A block indices (2x2 block of A for middle-dim block u)
            # A[0, 2u], A[0, 2u+1], A[1, 2u], A[1, 2u+1]
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            # B block indices (2x2 block of B for middle-dim block u, last-dim block v)
            # B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b11_idx = (2 * u) * 8 + 2 * v
            b12_idx = (2 * u) * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            # C block indices (W indices for the 2x2 output block)
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            # W index = k*2 + i
            c11_idx = (2 * v) * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = (2 * v) * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # Strassen's 7 products for 2x2 block multiplication
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # M1 = (A11+A22)(B11+B22)
            # C11 += M1, C22 += M1
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # M2 = (A21+A22)B11
            # C21 += M2, C22 -= M2
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # M3 = A11(B12-B22)
            # C12 += M3, C22 += M3
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # M4 = A22(B21-B11)
            # C11 += M4, C21 += M4
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # M5 = (A11+A12)B22
            # C11 -= M5, C12 += M5
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # M6 = (A21-A11)(B11+B12)
            # C22 += M6
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # M7 = (A12-A22)(B21+B22)
            # C11 += M7
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
    
    return {"U": U, "V": V, "W": W}
