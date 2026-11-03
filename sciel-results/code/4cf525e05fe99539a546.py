import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Rank 58 decomposition:
    # 1. Strassen 2x2x2 block for i,j,k in {0,1} (Rank 7)
    # 2. Schoolbook for j=2, k in {0,1}, i in {0,1,2} (Rank 6)
    # 3. Schoolbook for k in {2..6}, i in {0,1,2}, j in {0,1,2} (Rank 45)
    
    total_rank = 58
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Strassen vectors for 2x2x2 block (i,j in 0,1; k in 0,1)
    # Indices in U: i*3 + j
    # Indices in V: j*7 + k
    # Indices in W: k*3 + i
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[8, col] = 1
    W[0, col] = 1; W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # Term 2: M2 = (A10 + A11)B00
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1
    col += 1
    
    # Term 3: M3 = A00(B01 - B11)
    U[0, col] = 1
    V[1, col] = 1; V[8, col] = -1
    W[4, col] = -1
    col += 1
    
    # Term 4: M4 = A11(B10 - B00)
    U[4, col] = 1
    V[7, col] = 1; V[0, col] = -1
    W[1, col] = 1; W[0, col] = -1
    col += 1
    
    # Term 5: M5 = (A00 + A01)B11
    U[0, col] = 1; U[1, col] = 1
    V[8, col] = 1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # Term 6: M6 = (A10 - A00)(B00 + B01)
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[0, col] = -1
    col += 1
    
    # Term 7: M7 = (A01 - A11)(B10 + B11)
    U[1, col] = 1; U[4, col] = -1
    V[7, col] = 1; V[8, col] = 1
    W[0, col] = 1; W[4, col] = -1
    col += 1
    
    # Group 2: j=2, k in {0, 1}, i in {0, 1, 2}
    # Term: A[i, 2] * B[2, k] -> C[i, k]
    for i in range(3):
        for k in range(2):
            U[i * 3 + 2, col] = 1
            V[2 * 7 + k, col] = 1
            W[k * 3 + i, col] = 1
            col += 1
            
    # Group 3: k in {2..6}, i in {0, 1, 2}, j in {0, 1, 2}
    # Term: A[i, j] * B[j, k] -> C[i, k]
    for i in range(3):
        for j in range(3):
            for k in range(2, 7):
                U[i * 3 + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
