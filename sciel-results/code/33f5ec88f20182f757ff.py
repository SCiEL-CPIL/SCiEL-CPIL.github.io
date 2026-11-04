"""Rewritten program implementing a rank-33 tensor decomposition for 2x3x6 matrix multiplication.

Strategy: decompose into three independent 2x3x2 blocks (since p=6 = 3*2).
Each block uses Strassen for the 2x2 core (rank 7) + schoolbook for the outer product
with the third column/row (rank 4), giving 11 terms per block.
Total rank = 3 * 11 = 33.

The output W follows the convention W[k*n + i, r] where k is column of C (0..p-1), i is row of C (0..n-1).
"""

import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """Construct rank-33 tensor decomposition for 2x3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Each of the 3 blocks (b=0,1,2) handles columns 3b, 3b+1, 3b+2 of B and C
    # Each block has 11 terms: 7 Strassen + 4 schoolbook
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 33)
    V = [[0] * rank for _ in range(m * p)]  # (18, 33)
    W = [[0] * rank for _ in range(p * n)]  # (12, 33)
    
    # For each block b in {0, 1, 2}:
    # B columns: 3b, 3b+1, 3b+2 (call them col_a, col_b, col_c)
    # A uses all 3 columns: col 0, 1, 2
    # C columns: 3b, 3b+1, 3b+2
    
    # Strassen for 2x2 core (A cols 0,1; B rows 0,1):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[0,1])  -- wait, need to be careful
    # Actually for 2x2: A is 2x2 (rows 0,1; cols 0,1), B is 2x2 (rows 0,1; cols 0,1)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me redefine clearly:
    # A_2x2 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  (using cols 0,1 of A)
    # B_2x2 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]  (using rows 0,1 of B for this block)
    
    # Strassen terms:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Schoolbook for A col 2 * B row 2:
    # M8 = A[0,2] * B[2,0]  -> C[0,0] += M8
    # M9 = A[0,2] * B[2,1]  -> C[0,1] += M9
    # M10 = A[1,2] * B[2,0] -> C[1,0] += M10
    # M11 = A[1,2] * B[2,1] -> C[1,1] += M11
    
    for b in range(3):
        base_col = b * 11  # starting column index in U, V, W
        
        # B rows for this block: 3b, 3b+1, 3b+2
        # B row 0 in local = global row 3b
        # B row 1 in local = global row 3b+1
        # B row 2 in local = global row 3b+2
        
        # C columns for this block: 3b, 3b+1
        # C col 0 in local = global col 3b
        # C col 1 in local = global col 3b+1
        
        # --- Strassen term M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) ---
        # U: A[0,0] and A[1,1] contribute
        # A[0,0]: row 0*3+0=0, A[1,1]: row 1*3+1=4
        U[0][base_col + 0] = 1
        U[4][base_col + 0] = 1
        # V: B[0,0] and B[1,1] contribute
        # B[0,0]: row 0*6 + 3b, B[1,1]: row 1*6 + (3b+1)
        V[0 * 6 + 3 * b][base_col + 0] = 1
        V[1 * 6 + (3 * b + 1)][base_col + 0] = 1
        # W: C[0,0] += M1, C[1,1] += M1
        # C[0,0]: row (3b)*2 + 0 = 6b, C[1,1]: row (3b+1)*2 + 1 = 6b+3
        W[6 * b][base_col + 0] = 1
        W[6 * b + 3][base_col + 0] = 1
        
        # --- Strassen term M2: (A[1,0]+A[1,1]) * B[0,0] ---
        U[1 * 3 + 0][base_col + 1] = 1  # A[1,0]
        U[1 * 3 + 1][base_col + 1] = 1  # A[1,1]
        V[0 * 6 + 3 * b][base_col + 1] = 1  # B[0,0]
        # W: C[1,0] += M2, C[1,1] -= M2
        W[6 * b + 2][base_col + 1] = 1    # C[1,0]: row (3b)*2+1 = 6b+2
        W[6 * b + 3][base_col + 1] = -1   # C[1,1]: row (3b+1)*2+1 = 6b+3
        
        # --- Strassen term M3: A[0,0] * (B[0,1]-B[1,1]) ---
        U[0 * 3 + 0][base_col + 2] = 1  # A[0,0]
        V[0 * 6 + (3 * b + 1)][base_col + 2] = 1   # B[0,1]
        V[1 * 6 + (3 * b + 1)][base_col + 2] = -1  # B[1,1]
        # W: C[0,1] += M3, C[1,1] += M3
        W[6 * b + 1][base_col + 2] = 1   # C[0,1]: row (3b+1)*2+0 = 6b+2... wait
        # C[i, j]: i is row (0 or 1), j is col (3b or 3b+1)
        # W index: j * n + i = j * 2 + i
        # C[0,1] means row=0, col=3b+1: W index = (3b+1)*2 + 0 = 6b+2
        # C[1,1] means row=1, col=3b+1: W index = (3b+1)*2 + 1 = 6b+3
        W[6 * b + 2][base_col + 2] = 1   # C[0,1]
        W[6 * b + 3][base_col + 2] = 1   # C[1,1]
        
        # --- Strassen term M4: A[1,1] * (B[1,0]-B[0,0]) ---
        U[1 * 3 + 1][base_col + 3] = 1  # A[1,1]
        V[1 * 6 + 3 * b][base_col + 3] = 1     # B[1,0]
        V[0 * 6 + 3 * b][base_col + 3] = -1    # B[0,0]
        # W: C[0,0] += M4, C[1,0] += M4
        # C[0,0]: W index = 3b*2+0 = 6b
        # C[1,0]: W index = 3b*2+1 = 6b+1
        W[6 * b][base_col + 3] = 1
        W[6 * b + 1][base_col + 3] = 1
        
        # --- Strassen term M5: (A[0,0]+A[0,1]) * B[1,1] ---
        U[0 * 3 + 0][base_col + 4] = 1  # A[0,0]
        U[0 * 3 + 1][base_col + 4] = 1  # A[0,1]
        V[1 * 6 + (3 * b + 1)][base_col + 4] = 1  # B[1,1]
        # W: C[0,0] -= M5, C[0,1] += M5
        W[6 * b][base_col + 4] = -1     # C[0,0]
        W[6 * b + 2][base_col + 4] = 1  # C[0,1]
        
        # --- Strassen term M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]) ---
        U[1 * 3 + 0][base_col + 5] = 1   # A[1,0]
        U[0 * 3 + 0][base_col + 5] = -1  # A[0,0]
        V[0 * 6 + 3 * b][base_col + 5] = 1       # B[0,0]
        V[0 * 6 + (3 * b + 1)][base_col + 5] = 1 # B[0,1]
        # W: C[1,1] += M6
        W[6 * b + 3][base_col + 5] = 1   # C[1,1]
        
        # --- Strassen term M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) ---
        U[0 * 3 + 1][base_col + 6] = 1   # A[0,1]
        U[1 * 3 + 1][base_col + 6] = -1  # A[1,1]
        V[1 * 6 + 3 * b][base_col + 6] = 1       # B[1,0]
        V[1 * 6 + (3 * b + 1)][base_col + 6] = 1 # B[1,1]
        # W: C[0,0] += M7
        W[6 * b][base_col + 6] = 1       # C[0,0]
        
        # --- Schoolbook M8: A[0,2] * B[2,0] -> C[0,0] ---
        U[0 * 3 + 2][base_col + 7] = 1   # A[0,2]
        V[2 * 6 + 3 * b][base_col + 7] = 1  # B[2,0]
        W[6 * b][base_col + 7] = 1       # C[0,0]
        
        # --- Schoolbook M9: A[0,2] * B[2,1] -> C[0,1] ---
        U[0 * 3 + 2][base_col + 8] = 1   # A[0,2]
        V[2 * 6 + (3 * b + 1)][base_col + 8] = 1  # B[2,1]
        W[6 * b + 2][base_col + 8] = 1   # C[0,1]
        
        # --- Schoolbook M10: A[1,2] * B[2,0] -> C[1,0] ---
        U[1 * 3 + 2][base_col + 9] = 1   # A[1,2]
        V[2 * 6 + 3 * b][base_col + 9] = 1  # B[2,0]
        W[6 * b + 1][base_col + 9] = 1   # C[1,0]
        
        # --- Schoolbook M11: A[1,2] * B[2,1] -> C[1,1] ---
        U[1 * 3 + 2][base_col + 10] = 1  # A[1,2]
        V[2 * 6 + (3 * b + 1)][base_col + 10] = 1  # B[2,1]
        W[6 * b + 3][base_col + 10] = 1  # C[1,1]
    
    return {"U": U, "V": V, "W": W}
