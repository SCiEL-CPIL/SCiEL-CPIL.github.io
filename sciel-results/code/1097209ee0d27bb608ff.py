import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A indices: A11=(0,0), A22=(1,1) → U[0*2+0,0]=1, U[1*2+1,0]=1
    # B indices: B11=(0,0), B22=(1,1) → V[0*2+0,0]=1, V[1*2+1,0]=1
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    # W[k*2+i, r]: C11 → k=0,i=0 → W[0,0]; C12 → k=1,i=0 → W[2,0]; C21 → k=0,i=1 → W[1,0]; C22 → k=1,i=1 → W[3,0]
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = 1; W[3, 0] = 1  # C11, C12, C21, C22
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1  # B11
    W[1, 1] = 1; W[3, 1] = 1  # C21, C22
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1; V[3, 2] = -1  # B12, -B22
    W[2, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1; V[0, 3] = -1  # B21, -B11
    W[0, 3] = 1; W[1, 3] = 1  # C11, C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1; W[2, 4] = 1  # -C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1  # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1  # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1; W[3, 6] = 1  # C11, C22
    
    return U, V, W

def laderman_3x3():
    """Return U, V, W for 3x3 matrix multiplication using Laderman's algorithm.
    U has shape (9, 15), V has shape (9, 15), W has shape (9, 15).
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Laderman's 15 products for 3x3 matrix multiplication
    # This is complex. Let me use a known formulation.
    
    # Actually, let me use the schoolbook for 3x3 for now (rank 27) 
    # and improve later.
    
    # For now, return schoolbook
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Block decomposition: 2+3 split
    # A = [[A11(2x2), A12(2x3)], [A21(3x2), A22(3x3)]]
    # B = [[B11(2x2), B12(2x3)], [B21(3x2), B22(3x3)]]
    # C = [[C11(2x2), C12(2x3)], [C21(3x2), C22(3x3)]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Sub-problems and their ranks:
    # 1. A11*B11: 2x2, rank 7 (Strassen)
    # 2. A12*B21: 2x3 * 3x2 -> 2x2, rank 12 (schoolbook)
    # 3. A11*B12: 2x2 * 2x3 -> 2x3, rank 12 (schoolbook)
    # 4. A12*B22: 2x3 * 3x3 -> 2x3, rank 18 (schoolbook)
    # 5. A21*B11: 3x2 * 2x2 -> 3x2, rank 12 (schoolbook)
    # 6. A22*B21: 3x3 * 3x2 -> 3x2, rank 18 (schoolbook)
    # 7. A21*B12: 3x2 * 2x3 -> 3x3, rank 18 (schoolbook)
    # 8. A22*B22: 3x3, rank 27 (schoolbook) or 15 (Laderman)
    
    # Total rank: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    # Or with Laderman: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 15 = 112
    
    # Let's use Laderman for the 3x3 part
    # But I need to implement Laderman correctly.
    
    # For now, let's use schoolbook for all sub-problems to ensure correctness,
    # then optimize.
    
    # Actually, let me just use schoolbook for everything to get a valid decomposition,
    # then try to improve.
    
    # Schoolbook for the full 5x5 problem
    rank = n * m * p
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
