"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Rank 43 decomposition using Strassen for 2x2 sub-blocks
    # This exploits the structure:
    # C = A @ B
    # Split A into 2x2 + 1 rows, B into 2x5 + 1 row
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12, B13], [B21, B22, B23]] where B1* is 2x2, B2* is 2x2, B1* etc.
    # Actually let's use a cleaner block structure.
    
    # Let's use the following block decomposition:
    # A is 3x3, B is 3x5
    # Split rows of A: first 2 rows, last 1 row
    # Split columns of B: first 2 cols, next 2 cols, last 1 col
    
    # This gives 2 x 3 = 6 block multiplications:
    # (2x2)(2x2) + (2x2)(2x2) + (2x2)(2x1) + (1x2)(2x2) + (1x2)(2x2) + (1x2)(2x1)
    # = 2x2@2x2 + 2x2@2x2 + 2x2@2x1 + 1x2@2x2 + 1x2@2x2 + 1x2@2x1
    
    # Use Strassen for the two 2x2@2x2 blocks: 7 + 7 = 14
    # For 2x2@2x1: can use 4 mults (each output needs 2 mults)
    # For 1x2@2x2: 4 mults
    # For 1x2@2x1: 2 mults
    # Total: 14 + 4 + 4 + 2 = 24? No wait, that's not right.
    
    # Let me reconsider. The block structure gives:
    # C[0:2, 0:2] = A[0:2,0:2]@B[0:2,0:2] + A[0:2,2]@B[2,0:2]
    # C[0:2, 2:4] = A[0:2,0:2]@B[0:2,2:4] + A[0:2,2]@B[2,2:4]
    # C[0:2, 4]   = A[0:2,0:2]@B[0:2,4]   + A[0:2,2]@B[2,4]
    # C[2, 0:2]   = A[2,0:2]@B[0:2,0:2]   + A[2,2]@B[2,0:2]
    # C[2, 2:4]   = A[2,0:2]@B[0:2,2:4]   + A[2,2]@B[2,2:4]
    # C[2, 4]     = A[2,0:2]@B[0:2,4]     + A[2,2]@B[2,4]
    
    # This is getting complicated. Let me use a simpler approach:
    # Just use Strassen on the 2x2 corner and standard for the rest.
    
    # Actually, let me just implement the known rank-43 algorithm directly.
    # The key insight is that for 3x3x5, we can achieve rank 43 by combining
    # Strassen's 7-multiplication 2x2 algorithm with standard multiplications
    # for the remaining terms.
    
    # I'll construct the decomposition by enumerating all terms and then
    # applying Strassen where applicable.
    
    # For now, let's use a careful block decomposition:
    
    # Terms in A@B:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's define the 43 rank-1 tensors:
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Indices: A[i,j], B[j,k], C[i,k] with i in 0..2, j in 0..2, k in 0..4
    
    # Strassen for the 2x2 block (i in {0,1}, j in {0,1}, k in {0,1}):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # But B has 5 columns, so we need to handle k=0,1 separately from k=2,3,4.
    
    # Let me split B into two parts: B_left (cols 0,1) and B_right (cols 2,3,4)
    
    # For B_left (2 cols), use Strassen on the 2x2x2 part (i in {0,1}, j in {0,1}, k in {0,1})
    # Then handle the remaining terms.
    
    # This is getting complex. Let me just use the schoolbook 45 and try to optimize.
    
    # Actually, let me try a different approach: use the fact that we can share
    # some multiplications between adjacent blocks.
    
    # Let me just go with a clean implementation of rank 45 (schoolbook) 
    # and verify it works, then see if I can improve.
    
    rank = n * m * p  # 45
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

# EVOLVE-BLOCK-END
