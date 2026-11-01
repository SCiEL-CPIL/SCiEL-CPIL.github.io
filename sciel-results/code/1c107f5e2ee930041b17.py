import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x6) into B1 (2x6) and B2 (1x6)
    # C = A1 B1 + A2 B2
    # A1 B1 is 2x2x6. Split B1 into three 2x2 blocks. Use Strassen (rank 7) for each.
    # A2 B2 is 2x1x6. Use schoolbook (rank 12).
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6x33
    V = [[0] * rank for _ in range(m * p)] # 18x33
    W = [[0] * rank for _ in range(p * n)] # 12x33
    
    # Strassen vectors for 2x2x2
    # U indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # V indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # W indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2) -> k*n+i
    # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A11, A22
        [1, 0, 1, 1], # M2: A21, A22 (Wait, M2 is (A21+A22)B11. So A21, A22)
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11, A12
        [1, 0, 1, 0], # M6: A21, A11 (-1) -> A21 - A11. So A11=-1, A21=1.
        [0, 1, 0, 1]  # M7: A12, A22 (-1) -> A12 - A22. So A12=1, A22=-1.
    ]
    # Correcting Strassen U vectors based on standard definitions:
    # M1: (A11+A22) -> [1, 0, 0, 1]
    # M2: (A21+A22) -> [0, 0, 1, 1]
    # M3: A11 -> [1, 0, 0, 0]
    # M4: A22 -> [0, 0, 0, 1]
    # M5: (A11+A12) -> [1, 1, 0, 0]
    # M6: (A21-A11) -> [-1, 0, 1, 0]
    # M7: (A12-A22) -> [0, 1, 0, -1]
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: C11, C22
        [0, 0, 1, -1],# M2: C21, -C22
        [0, 1, 0, 1], # M3: C12, C22
        [1, 0, 1, 0], # M4: C11, C21
        [0, 1, 0, -1],# M5: C12, -C22
        [0, 0, 0, 1], # M6: C22
        [1, 0, 0, 0]  # M7: C11
    ]
    
    col = 0
    # 3 Strassen blocks for A1 (2x2) * B1 (2x6)
    # B1 is rows 0,1 of B. Cols 0..5.
    # Block 0: Cols 0,1 of B1.
    # Block 1: Cols 2,3 of B1.
    # Block 2: Cols 4,5 of B1.
    
    for block in range(3):
        for s in range(7):
            # U: A1 is rows 0,1 of A. Cols 0,1.
            # U index: i*m + j. i in 0,1. j in 0,1.
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = strassen_U[s][i * 2 + j]
            
            # V: B1 is rows 0,1 of B. Cols 2*block, 2*block+1.
            # V index: j*p + k. j in 0,1. k in 2*block, 2*block+1.
            for j in range(2):
                for k in range(2):
                    V[j * p + (2 * block + k)][col] = strassen_V[s][j * 2 + k]
            
            # W: C is rows 0,1 of C (since A1 is 2x2). Cols 2*block, 2*block+1.
            # W index: k*n + i. k in 2*block, 2*block+1. i in 0,1.
            for k in range(2):
                for i in range(2):
                    W[(2 * block + k) * n + i][col] = strassen_W[s][k * 2 + i]
            
            col += 1
    
    # Outer product for A2 (2x1) * B2 (1x6)
    # A2 is row 0,1 of A. Col 2.
    # B2 is row 2 of B. Cols 0..5.
    # C2 is rows 0,1 of C. Cols 0..5.
    
    for i in range(2):
        for k in range(6):
            # U: A[i, 2]
            U[i * m + 2][col] = 1
            # V: B[2, k]
            V[2 * p + k][col] = 1
            # W: C[k, i] -> index k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
