import time
import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Use block Strassen: split A into 2x2 blocks of 2x2, B into 2x3 blocks of 2x3
    # 7 Strassen multiplications, each 2x2 * 2x3 (rank 12 schoolbook)
    # Total rank: 7 * 12 = 84
    
    # A indices (row-major 4x4): A[i,k] at i*4+k
    # A11: A[0,0],A[0,1],A[1,0],A[1,1] = idx 0,1,4,5
    # A12: A[0,2],A[0,3],A[1,2],A[1,3] = idx 2,3,6,7
    # A21: A[2,0],A[2,1],A[3,0],A[3,1] = idx 8,9,12,13
    # A22: A[2,2],A[2,3],A[3,2],A[3,3] = idx 10,11,14,15
    
    # B indices (row-major 4x9): B[k,j] at k*9+j
    # B11: B[0,0],B[0,1],B[0,2],B[1,0],B[1,1],B[1,2] = idx 0,1,2,9,10,11
    # B12: B[0,3],B[0,4],B[0,5],B[1,3],B[1,4],B[1,5] = idx 3,4,5,12,13,14
    # B13: B[0,6],B[0,7],B[0,8],B[1,6],B[1,7],B[1,8] = idx 6,7,8,15,16,17
    # B21: B[2,0],B[2,1],B[2,2],B[3,0],B[3,1],B[3,2] = idx 18,19,20,27,28,29
    # B22: B[2,3],B[2,4],B[2,5],B[3,3],B[3,4],B[3,5] = idx 21,22,23,30,31,32
    # B23: B[2,6],B[2,7],B[2,8],B[3,6],B[3,7],B[3,8] = idx 24,25,26,33,34,35
    
    # C indices (W convention: C[i,j] at j*4+i)
    # C11: C[0,0],C[0,1],C[1,0],C[1,1] -> W[0],W[1],W[4],W[5]
    # C12: C[0,2],C[0,3],C[1,2],C[1,3] -> W[8],W[9],W[12],W[13]
    # C13: C[0,4..8],C[1,4..8] -> W[16,17,20,21,24,25,28,29,32,33]
    # C21: C[2,0],C[2,1],C[3,0],C[3,1] -> W[2,3,6,7]
    # C22: C[2,2],C[2,3],C[3,2],C[3,3] -> W[10,11,14,15]
    # C23: C[2,4..8],C[3,4..8] -> W[18,19,22,23,26,27,30,31,34,35]
    
    # Strassen 7 multiplications for 2x2 block structure:
    # M1 = (A11+A22) * (B11+B21)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B13-B23)
    # M5 = (A11+A12) * B23
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # C11 = M1 - M2 - M6 + M7
    # C12 = M1 - M2 - M4 - M6  (need to verify)
    # C13 = M3 + M4 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 - M2 - M6 (need to verify)
    # C23 = M3 + M4 + M5 - M7
    
    # Let me verify these formulas by expanding:
    # M1 = A11B11 + A11B21 + A22B11 + A22B21
    # M2 = A21B11 + A22B11
    # M3 = A11B12 - A11B22
    # M4 = A22B13 - A22B23
    # M5 = A11B23 + A12B23
    # M6 = A21B11 + A21B12 - A11B11 - A11B12
    # M7 = A12B21 + A12B22 - A22B21 - A22B22
    
    # C11 = A11B11 + A12B21
    # M1 - M2 - M6 + M7 = (A11B11+A11B21+A22B11+A22B21) - (A21B11+A22B11) - (A21B11+A21B12-A11B11-A11B12) + (A12B21+A12B22-A22B21-A22B22)
    # = A11B11+A11B21+A22B11+A22B21 - A21B11-A22B11 - A21B11-A21B12+A11B11+A11B12 + A12B21+A12B22-A22B21-A22B22
    # = 2A11B11 + A11B21 + A11B12 + A12B21 + A12B22 - 2A21B11 - A21B12 - 2A22B22
    # This is NOT equal to A11B11 + A12B21. So the formulas are wrong for rectangular case.
    
    # The standard Strassen doesn't directly apply to rectangular blocks this way.
    # Let me just use schoolbook rank 144 which is correct.
    
    rank = 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
