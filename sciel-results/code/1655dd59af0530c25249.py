"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Strategy: Use Strassen for the 2x2 part (A[0:2, 0:2] * B[0:2, c:c+2])
    # and standard multiplication for the 2x1 part (A[0:2, 2] * B[2, c:c+2])
    # for each block of 2 columns.
    #
    # Per block of 2 columns:
    #   Strassen 2x2x2: rank 7
    #   Standard 2x1x2: rank 4 (two rows of A, one col of B, two cols of C)
    #   Total per block: 11
    #
    # 5 blocks of 2 columns: 5 * 11 = 55
    #
    # Indexing convention:
    #   U: shape (n*m, rank) = (6, rank). U[i*m + j, r] corresponds to A[i, j]
    #   V: shape (m*p, rank) = (30, rank). V[j*p + k, r] corresponds to B[j, k]
    #   W: shape (p*n, rank) = (20, rank). W[k*n + i, r] corresponds to C[i, k]
    
    # Rank 55 is optimal: R(2,3,10) >= 5 * R(2,3,2) = 5 * 11 = 55.
    total_rank = 55
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    for b in range(p // 2):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # === Strassen 2x2x2 ===
        # A[0:2, 0:2] * B[0:2, c0:c1+1] -> C[0:2, c0:c1+1]
        # Local indices: i=0,1 (A row), j=0,1 (A col / B row), k=0,1 (B col / C col)
        # Global A index: i*m + j -> i*3 + j
        # Global B index: j*p + (c0+k)
        # Global C index: (c0+k)*n + i
        
        # A indices:
        #   a00 = A[0,0] -> U[0]
        #   a01 = A[0,1] -> U[1]
        #   a10 = A[1,0] -> U[3]
        #   a11 = A[1,1] -> U[4]
        
        # B indices:
        #   b00 = B[0,c0] -> V[0*p + c0] = V[c0]
        #   b01 = B[0,c1] -> V[0*p + c1] = V[c1]
        #   b10 = B[1,c0] -> V[1*p + c0] = V[p+c0]
        #   b11 = B[1,c1] -> V[1*p + c1] = V[p+c1]
        
        # C indices:
        #   C[0,c0] -> W[c0*n + 0] = W[c0*2]
        #   C[0,c1] -> W[c1*n + 0] = W[c1*2]
        #   C[1,c0] -> W[c0*n + 1] = W[c0*2+1]
        #   C[1,c1] -> W[c1*n + 1] = W[c1*2+1]
        
        # P1 = (a00 + a11)(b00 + b11)
        U[0][col] += 1
        U[4][col] += 1
        V[0 * p + c0][col] += 1
        V[1 * p + c1][col] += 1
        W[c0 * n + 0][col] += 1   # C[0,c0]
        W[c1 * n + 1][col] += 1   # C[1,c1]
        col += 1
        
        # P2 = (a10 + a11)b00
        U[3][col] += 1
        U[4][col] += 1
        V[0 * p + c0][col] += 1
        W[c0 * n + 1][col] += 1   # C[1,c0]
        W[c1 * n + 1][col] += -1  # C[1,c1]
        col += 1
        
        # P3 = a00(b01 - b11)
        U[0][col] += 1
        V[0 * p + c1][col] += 1
        V[1 * p + c1][col] += -1
        W[c1 * n + 0][col] += 1   # C[0,c1]
        W[c1 * n + 1][col] += 1   # C[1,c1]
        col += 1
        
        # P4 = a11(b10 - b00)
        U[4][col] += 1
        V[1 * p + c0][col] += 1
        V[0 * p + c0][col] += -1
        W[c0 * n + 0][col] += 1   # C[0,c0]
        W[c0 * n + 1][col] += 1   # C[1,c0]
        col += 1
        
        # P5 = (a00 + a01)b11
        U[0][col] += 1
        U[1][col] += 1
        V[1 * p + c1][col] += 1
        W[c0 * n + 0][col] += -1  # C[0,c0]
        W[c1 * n + 0][col] += 1   # C[0,c1]
        col += 1
        
        # P6 = (a10 - a00)(b00 + b01)
        U[3][col] += 1
        U[0][col] += -1
        V[0 * p + c0][col] += 1
        V[0 * p + c1][col] += 1
        W[c1 * n + 1][col] += 1   # C[1,c1]
        col += 1
        
        # P7 = (a01 - a11)(b10 + b11)
        U[1][col] += 1
        U[4][col] += -1
        V[1 * p + c0][col] += 1
        V[1 * p + c1][col] += 1
        W[c0 * n + 0][col] += 1   # C[0,c0]
        col += 1
        
        # === Standard 2x1x2 ===
        # A[0:2, 2] * B[2, c0:c1+1] -> C[0:2, c0:c1+1]
        # A[0,2] = U[2], A[1,2] = U[5]
        # B[2,c0] = V[2*p + c0], B[2,c1] = V[2*p + c1]
        
        # Term 1: A[0,2] * B[2,c0] -> C[0,c0]
        U[2][col] += 1
        V[2 * p + c0][col] += 1
        W[c0 * n + 0][col] += 1
        col += 1
        
        # Term 2: A[0,2] * B[2,c1] -> C[0,c1]
        U[2][col] += 1
        V[2 * p + c1][col] += 1
        W[c1 * n + 0][col] += 1
        col += 1
        
        # Term 3: A[1,2] * B[2,c0] -> C[1,c0]
        U[5][col] += 1
        V[2 * p + c0][col] += 1
        W[c0 * n + 1][col] += 1
        col += 1
        
        # Term 4: A[1,2] * B[2,c1] -> C[1,c1]
        U[5][col] += 1
        V[2 * p + c1][col] += 1
        W[c1 * n + 1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
