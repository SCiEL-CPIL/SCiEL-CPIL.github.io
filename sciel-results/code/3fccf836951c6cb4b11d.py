"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Rank 39 decomposition using Strassen for 2x2x2 blocks
    # A is 2x3, B is 3x7, C is 2x7
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (2x7) and B2 (1x7)
    # C = A1 B1 + A2 B2
    
    # Part 1: A2 B2 (Rank 14)
    # A2 indices: i in {0,1}, j=2
    # B2 indices: j=2, k in {0..6}
    # C indices: i in {0,1}, k in {0..6}
    
    # Part 2: A1 B1 (Rank 25)
    # A1 indices: i in {0,1}, j in {0,1}
    # B1 indices: j in {0,1}, k in {0..6}
    # Split B1 into B1_0 (2x2), B1_1 (2x2), B1_2 (2x2), B1_3 (2x1)
    # B1_0 cols 0,1; B1_1 cols 2,3; B1_2 cols 4,5; B1_3 col 6
    
    # Total Rank = 14 + 3*7 + 4 = 39
    
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A2 B2
    # 14 terms
    for i in range(n):
        for k in range(p):
            # U index: i*m + 2
            U[i * m + 2, col] = 1
            # V index: 2*p + k
            V[2 * p + k, col] = 1
            # W index: k*n + i
            W[k * n + i, col] = 1
            col += 1
            
    # Part 2: A1 B1
    # Strassen for 3 blocks of 2x2x2
    # Strassen coefficients for M_{2,2,2}
    # U, V, W for each of the 7 products
    
    # Strassen terms (U, V, W)
    # U is 2x2, V is 2x2, W is 2x2
    # Indices in U: i*2+j, V: j*2+k, W: k*2+i
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0],# M6: A10, -A00
        [-1, 0, 0, 1] # M7: A11, -A01
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1: B00, B11
        [0, 1, 0, 0], # M2: B01
        [0, 0, 1, -1],# M3: B10, -B11
        [1, 0, -1, 0],# M4: B00, -B10
        [0, 0, 0, 1], # M5: B11
        [1, 0, 1, 0], # M6: B00, B10
        [0, 1, 0, 1]  # M7: B01, B11
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, 0], # M2: C01
        [0, 0, 0, 0], # M3: C01? No, M3 contributes to C01
        # Let's check Strassen formulas again
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3
        # C10 = M6 + M5
        # C11 = M1 - M2 + M4 + M7
        
        # W should be the coefficients of M_s in C_ik
        # W index k*2+i
        # M1: C00 (1), C11 (1) -> W[0*2+0]=1, W[1*2+1]=1
        # M2: C01 (1) -> W[1*2+0]=1
        # M3: C01 (1) -> W[1*2+0]=1
        # M4: C00 (1), C11 (1) -> W[0*2+0]=1, W[1*2+1]=1
        # M5: C00 (-1), C10 (1) -> W[0*2+0]=-1, W[0*2+1]=1
        # M6: C10 (1) -> W[0*2+1]=1
        # M7: C00 (1), C11 (1) -> W[0*2+0]=1, W[1*2+1]=1
    ]
    
    # Correct W definitions:
    # W is a vector of length 4 (2x2 matrix)
    # Index k*2 + i
    strassen_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, 0], # M2: C01
        [0, 1, 0, 0], # M3: C01
        [1, 0, 0, 1], # M4: C00, C11
        [-1, 1, 0, 0],# M5: -C00, C10
        [0, 1, 0, 0], # M6: C10
        [1, 0, 0, 1]  # M7: C00, C11
    ]
    
    # Apply Strassen to 3 blocks
    for block_idx in range(3):
        # Block cols: 2*block_idx, 2*block_idx+1
        k_offset = 2 * block_idx
        
        for s in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] += strassen_U[s][i * 2 + j]
                    
            # V
            for j in range(2):
                for k in range(2):
                    # Global k index: k_offset + k
                    V[j * p + (k_offset + k), col] += strassen_V[s][j * 2 + k]
                    
            # W
            for k in range(2):
                for i in range(2):
                    # Global k index: k_offset + k
                    W[(k_offset + k) * n + i, col] += strassen_W[s][k * 2 + i]
                    
            col += 1
            
    # Part 3: A1 B1_3 (2x2 x 2x1)
    # B1_3 is col 6
    # Schoolbook for 2x2 x 2x1 (Rank 4)
    for i in range(2):
        for j in range(2):
            U[i * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
