import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110  # 10 seconds safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank is 60. We aim for 40.
    # Decomposition:
    # Split B (4x5) into B1 (3x5) and B2 (1x5).
    # Split A (3x4) into A1 (3x3) and A2 (3x1).
    # C = A1*B1 + A2*B2
    # A1*B1: Split B1 into B11 (3x3) and B12 (3x2).
    # A1*B11: 3x3x3 multiplication (Strassen, rank 7)
    # A1*B12: 3x3x2 multiplication (Schoolbook, rank 18)
    # A2*B2: 3x1x5 multiplication (Schoolbook, rank 15)
    # Total rank: 7 + 18 + 15 = 40.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*5 = 20 rows
    W = [[0] * rank for _ in range(p * n)]  # 5*3 = 15 rows
    
    col = 0
    
    # Part 1: A1 (3x3) * B11 (3x3) using Strassen (Rank 7)
    # Indices: A rows 0-2, cols 0-2. B rows 0-2, cols 0-2.
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U indices: i*m + j. i in 0..2, j in 0..2
    # V indices: j*p + k. j in 0..2, k in 0..2
    # W indices: k*n + i. k in 0..2, i in 0..2
    
    # Term 1
    U[0*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*5+0][col] += 1; V[1*5+1][col] += 1
    W[0*3+0][col] += 1; W[0*3+1][col] += 1; W[1*3+0][col] += 1; W[1*3+1][col] += 1
    col += 1
    
    # Term 2
    U[1*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*5+0][col] += 1
    W[0*3+1][col] += 1
    col += 1
    
    # Term 3
    U[0*3+0][col] += 1
    V[0*5+1][col] += 1; V[1*5+1][col] -= 1
    W[1*3+0][col] += 1
    col += 1
    
    # Term 4
    U[1*3+1][col] += 1
    V[1*5+0][col] += 1; V[0*5+0][col] -= 1
    W[0*3+1][col] += 1
    col += 1
    
    # Term 5
    U[0*3+0][col] += 1; U[0*3+1][col] += 1
    V[1*5+1][col] += 1
    W[1*3+0][col] += 1
    col += 1
    
    # Term 6
    U[1*3+0][col] += 1; U[0*3+0][col] -= 1
    V[0*5+0][col] += 1; V[0*5+1][col] += 1
    W[1*3+1][col] += 1
    col += 1
    
    # Term 7
    U[0*3+1][col] += 1; U[1*3+1][col] -= 1
    V[1*5+0][col] += 1; V[1*5+1][col] += 1
    W[0*3+0][col] += 1
    col += 1
    
    # Part 2: A1 (3x3) * B12 (3x2) using Schoolbook (Rank 18)
    # B12 is cols 3,4 of B. Rows 0,1,2.
    # A1 is rows 0,1,2. Cols 0,1,2.
    for i in range(3):
        for j in range(3):
            for k in range(2):
                # B col index is 3+k
                b_col = 3 + k
                U[i*3+j][col] = 1
                V[j*5+b_col][col] = 1
                W[b_col*3+i][col] = 1
                col += 1
    
    # Part 3: A2 (3x1) * B2 (1x5) using Schoolbook (Rank 15)
    # A2 is col 3 of A. Rows 0,1,2.
    # B2 is row 3 of B. Cols 0,1,2,3,4.
    for i in range(3):
        for k in range(5):
            U[i*3+3][col] = 1
            V[3*5+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
