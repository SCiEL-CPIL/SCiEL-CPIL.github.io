"""Matrix multiplication 3x3x7 - tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    The decomposition has the form:
    For each (i,j,k) where i in [0,n), j in [0,m), k in [0,p):
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Rank 26 decomposition for 3x3x3 part (k=0,1,2)
    # Rank 36 decomposition for 3x3x4 part (k=3,4,5,6)
    rank_3x3x3 = 26
    rank_3x3x4 = 36
    total_rank = rank_3x3x3 + rank_3x3x4
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # --- Part 1: 3x3x3 (k=0,1,2) ---
    # Indices for A (9x1), B (9x1), C (9x1) in local 3x3x3 space
    # Global U index: i*3+j
    # Global V index: j*7+k
    # Global W index: k*3+i
    
    # Strassen 2x2x2 terms for A00*B00
    # A00: [[A00, A01], [A10, A11]] -> U indices: 0, 1, 3, 4
    # B00: [[B00, B01], [B10, B11]] -> V indices: 0, 1, 3, 4 (k=0,1)
    # C00: [[C00, C01], [C10, C11]] -> W indices: 0, 1, 3, 4 (k=0,1)
    
    r = 0
    # P1 = (A00+A11)(B00+B11)
    U[0, r] = 1; U[4, r] = 1
    V[0, r] = 1; V[4, r] = 1
    W[0, r] = 1; W[4, r] = 1
    r += 1
    
    # P2 = (A10+A11)B00
    U[3, r] = 1; U[4, r] = 1
    V[0, r] = 1
    W[3, r] = 1; W[4, r] = 1
    r += 1
    
    # P3 = A00(B10-B11)
    U[0, r] = 1
    V[3, r] = 1; V[4, r] = -1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # P4 = A11(B01-B00)
    U[4, r] = 1
    V[1, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # P5 = (A00+A01)B11
    U[0, r] = 1; U[1, r] = 1
    V[4, r] = 1
    W[0, r] = -1; W[1, r] = 1
    r += 1
    
    # P6 = (A10-A00)(B00+B01)
    U[3, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[4, r] = 1
    r += 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1, r] = 1; U[4, r] = -1
    V[3, r] = 1; V[4, r] = 1
    W[0, r] = 1; W[1, r] = 1
    r += 1
    
    # Terms for A01*B10 (2x1 * 1x2 -> 2x2 added to C00)
    # A01: [A02, A12]^T -> U indices: 2, 5
    # B10: [B20, B21] -> V indices: 6, 7 (k=2, j=0,1 -> V[0*7+2]=2? No. V[j*7+k]. j=2, k=0 -> 14. j=2, k=1 -> 15.)
    # Wait, V index is j*p+k.
    # For k=0,1,2:
    # B is 3x3. B[j,k].
    # V index for B[j,k] is j*7+k.
    # B[2,0] -> V[14]. B[2,1] -> V[15].
    # A is 3x3. A[i,j].
    # U index for A[i,j] is i*3+j.
    # A[0,2] -> U[2]. A[1,2] -> U[5].
    # C is 3x3. C[i,k].
    # W index for C[i,k] is k*3+i.
    # C[0,0] -> W[0]. C[0,1] -> W[1]. C[1,0] -> W[3]. C[1,1] -> W[4].
    
    # T1: A[0,2] * B[2,0] -> C[0,0]
    U[2, r] = 1
    V[14, r] = 1
    W[0, r] = 1
    r += 1
    
    # T2: A[0,2] * B[2,1] -> C[0,1]
    U[2, r] = 1
    V[15, r] = 1
    W[1, r] = 1
    r += 1
    
    # T3: A[1,2] * B[2,0] -> C[1,0]
    U[5, r] = 1
    V[14, r] = 1
    W[3, r] = 1
    r += 1
    
    # T4: A[1,2] * B[2,1] -> C[1,1]
    U[5, r] = 1
    V[15, r] = 1
    W[4, r] = 1
    r += 1
    
    # Terms for A00*B01 (2x2 * 2x1 -> 2x1 added to C01)
    # A00: U[0,1,3,4]
    # B01: B[0,2], B[1,2] -> V[2], V[9]
    # C01: C[0,2], C[1,2] -> W[6], W[7]
    
    # T5: A[0,0] * B[0,2] -> C[0,2]
    U[0, r] = 1
    V[2, r] = 1
    W[6, r] = 1
    r += 1
    
    # T6: A[0,1] * B[1,2] -> C[0,2]
    U[1, r] = 1
    V[9, r] = 1
    W[6, r] = 1
    r += 1
    
    # T7: A[1,0] * B[0,2] -> C[1,2]
    U[3, r] = 1
    V[2, r] = 1
    W[7, r] = 1
    r += 1
    
    # T8: A[1,1] * B[1,2] -> C[1,2]
    U[4, r] = 1
    V[9, r] = 1
    W[7, r] = 1
    r += 1
    
    # Terms for A01*B11 (2x1 * 1x1 -> 2x1 added to C01)
    # A01: U[2,5]
    # B11: B[2,2] -> V[16]
    # C01: W[6,7]
    
    # T9: A[0,2] * B[2,2] -> C[0,2]
    U[2, r] = 1
    V[16, r] = 1
    W[6, r] = 1
    r += 1
    
    # T10: A[1,2] * B[2,2] -> C[1,2]
    U[5, r] = 1
    V[16, r] = 1
    W[7, r] = 1
    r += 1
    
    # Terms for A10*B00 (1x2 * 2x2 -> 1x2 added to C10)
    # A10: A[2,0], A[2,1] -> U[6,7]
    # B00: B[0,0], B[0,1], B[1,0], B[1,1] -> V[0,1,7,8]
    # C10: C[2,0], C[2,1] -> W[2], W[5]
    
    # T11: A[2,0] * B[0,0] -> C[2,0]
    U[6, r] = 1
    V[0, r] = 1
    W[2, r] = 1
    r += 1
    
    # T12: A[2,1] * B[1,0] -> C[2,0]
    U[7, r] = 1
    V[7, r] = 1
    W[2, r] = 1
    r += 1
    
    # T13: A[2,0] * B[0,1] -> C[2,1]
    U[6, r] = 1
    V[1, r] = 1
    W[5, r] = 1
    r += 1
    
    # T14: A[2,1] * B[1,1] -> C[2,1]
    U[7, r] = 1
    V[8, r] = 1
    W[5, r] = 1
    r += 1
    
    # Terms for A11*B10 (1x1 * 1x2 -> 1x2 added to C10)
    # A11: A[2,2] -> U[8]
    # B10: B[2,0], B[2,1] -> V[14,15]
    # C10: W[2,5]
    
    # T15: A[2,2] * B[2,0] -> C[2,0]
    U[8, r] = 1
    V[14, r] = 1
    W[2, r] = 1
    r += 1
    
    # T16: A[2,2] * B[2,1] -> C[2,1]
    U[8, r] = 1
    V[15, r] = 1
    W[5, r] = 1
    r += 1
    
    # Terms for A10*B01 (1x2 * 2x1 -> 1x1 added to C11)
    # A10: U[6,7]
    # B01: V[2,9]
    # C11: C[2,2] -> W[8]
    
    # T17: A[2,0] * B[0,2] -> C[2,2]
    U[6, r] = 1
    V[2, r] = 1
    W[8, r] = 1
    r += 1
    
    # T18: A[2,1] * B[1,2] -> C[2,2]
    U[7, r] = 1
    V[9, r] = 1
    W[8, r] = 1
    r += 1
    
    # Terms for A11*B11 (1x1 * 1x1 -> 1x1 added to C11)
    # A11: U[8]
    # B11: V[16]
    # C11: W[8]
    
    # T19: A[2,2] * B[2,2] -> C[2,2]
    U[8, r] = 1
    V[16, r] = 1
    W[8, r] = 1
    r += 1
    
    # --- Part 2: 3x3x4 (k=3,4,5,6) ---
    # Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
                r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
