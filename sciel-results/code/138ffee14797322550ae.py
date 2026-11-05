import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    
    # Rank 23 Decomposition
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Term 0: A00*B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # Term 1: A11*B11
    U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
    
    # Term 2: A22*B22
    U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
    
    # Term 3: (A01+A02)*(B10+B20)
    U[1, 3] = 1; U[2, 3] = 1
    V[3, 3] = 1; V[6, 3] = 1
    W[0, 3] = 0.5
    
    # Term 4: (A01-A02)*(B10-B20)
    U[1, 4] = 1; U[2, 4] = -1
    V[3, 4] = 1; V[6, 4] = -1
    W[0, 4] = 0.5
    
    # Term 5: (A10+A20)*(B01+B02)
    U[3, 5] = 1; U[6, 5] = 1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 0.5 # C[1,0] index 3? No. C[i,k]. i=1, k=0 -> index 3.
    # Check: U[3] is A[1,0]. V[1] is B[0,1]. j=0 matches. Gamma = k*3+i = 1*3+1 = 4? No.
    # U index 3: i=1, j=0.
    # V index 1: j=0, k=1.
    # Gamma = k*3+i = 1*3+1 = 4. C[1,1]? No.
    # C[i,k]. i=1, k=1.
    # Wait. A[1,0]*B[0,1] contributes to C[1,1].
    # A[1,0]*B[0,2] contributes to C[1,2].
    # A[2,0]*B[0,1] contributes to C[2,1].
    # A[2,0]*B[0,2] contributes to C[2,2].
    # So Term 5 contributes to C[1,1], C[1,2], C[2,1], C[2,2].
    # Indices:
    # C[1,1] -> 1*3+1 = 4.
    # C[1,2] -> 2*3+1 = 7.
    # C[2,1] -> 1*3+2 = 5.
    # C[2,2] -> 2*3+2 = 8.
    # So W[4,5], W[7,5], W[5,5], W[8,5] should be 0.5.
    W[4, 5] = 0.5
    W[7, 5] = 0.5
    W[5, 5] = 0.5
    W[8, 5] = 0.5
    
    # Term 6: (A10-A20)*(B01-B02)
    U[3, 6] = 1; U[6, 6] = -1
    V[1, 6] = 1; V[2, 6] = -1
    W[4, 6] = 0.5
    W[7, 6] = -0.5 # C[1,2]
    W[5, 6] = 0.5 # C[2,1]
    W[8, 6] = -0.5 # C[2,2]
    
    # Term 7: (A12+A21)*(B21+B12)
    # A[1,2], A[2,1]. B[2,1], B[1,2].
    # Pairs:
    # A[1,2]*B[2,1] -> C[1,1]. (j=2 matches).
    # A[1,2]*B[1,2] -> Mismatch j=2 vs 1.
    # A[2,1]*B[2,1] -> Mismatch j=1 vs 2.
    # A[2,1]*B[1,2] -> C[2,2]. (j=1 matches).
    U[5, 7] = 1; U[7, 7] = 1
    V[7, 7] = 1; V[5, 7] = 1
    W[4, 7] = 0.5 # C[1,1]
    W[8, 7] = 0.5 # C[2,2]
    
    # Term 8: (A12-A21)*(B21-B12)
    U[5, 8] = 1; U[7, 8] = -1
    V[7, 8] = 1; V[5, 8] = -1
    W[4, 8] = 0.5
    W[8, 8] = -0.5
    
    # Terms 9-14: Diagonal cross terms
    # Term 9: (A00+A11)*(B00+B11)
    # A00*B00 (C00), A00*B11 (C01? No, j=0 vs 1 mismatch), A11*B00 (C10? j=1 vs 0 mismatch), A11*B11 (C11).
    # So only C00 and C11.
    U[0, 9] = 1; U[4, 9] = 1
    V[0, 9] = 1; V[4, 9] = 1
    W[0, 9] = 0.5
    W[4, 9] = 0.5
    
    # Term 10: (A00-A11)*(B00-B11)
    U[0, 10] = 1; U[4, 10] = -1
    V[0, 10] = 1; V[4, 10] = -1
    W[0, 10] = 0.5
    W[4, 10] = 0.5
    
    # Term 11: (A11+A22)*(B11+B22)
    U[4, 11] = 1; U[8, 11] = 1
    V[4, 11] = 1; V[8, 11] = 1
    W[4, 11] = 0.5
    W[8, 11] = 0.5
    
    # Term 12: (A11-A22)*(B11-B22)
    U[4, 12] = 1; U[8, 12] = -1
    V[4, 12] = 1; V[8, 12] = -1
    W[4, 12] = 0.5
    W[8, 12] = 0.5
    
    # Term 13: (A00+A22)*(B00+B22)
    U[0, 13] = 1; U[8, 13] = 1
    V[0, 13] = 1; V[8, 13] = 1
    W[0, 13] = 0.5
    W[8, 13] = 0.5
    
    # Term 14: (A00-A22)*(B00-B22)
    U[0, 14] = 1; U[8, 14] = -1
    V[0, 14] = 1; V[8, 14] = -1
    W[0, 14] = 0.5
    W[8, 14] = 0.5
    
    # Terms 15-22: Off-diagonal cross terms
    # Term 15: (A01+A10)*(B10+B00)
    # A01*B10 (C00), A01*B00 (Mismatch), A10*B10 (Mismatch), A10*B00 (C10).
    U[1, 15] = 1; U[3, 15] = 1
    V[3, 15] = 1; V[0, 15] = 1
    W[0, 15] = 0.5
    W[3, 15] = 0.5
    
    # Term 16: (A01-A10)*(B10-B00)
    U[1, 16] = 1; U[3, 16] = -1
    V[3, 16] = 1; V[0, 16] = -1
    W[0, 16] = 0.5
    W[3, 16] = -0.5
    
    # Term 17: (A02+A20)*(B20+B00)
    # A02*B20 (C00), A02*B00 (Mismatch), A20*B20 (Mismatch), A20*B00 (C20).
    U[2, 17] = 1; U[6, 17] = 1
    V[6, 17] = 1; V[0, 17] = 1
    W[0, 17] = 0.5
    W[6, 17] = 0.5
    
    # Term 18: (A02-A20)*(B20-B00)
    U[2, 18] = 1; U[6, 18] = -1
    V[6, 18] = 1; V[0, 18] = -1
    W[0, 18] = 0.5
    W[6, 18] = -0.5
    
    # Term 19: (A12+A21)*(B12+B21)
    # A12*B12 (Mismatch), A12*B21 (C11), A21*B12 (C22), A21*B21 (Mismatch).
    U[5, 19] = 1; U[7, 19] = 1
    V[5, 19] = 1; V[7, 19] = 1
    W[4, 19] = 0.5
    W[8, 19] = 0.5
    
    # Term 20: (A12-A21)*(B12-B21)
    U[5, 20] = 1; U[7, 20] = -1
    V[5, 20] = 1; V[7, 20] = -1
    W[4, 20] = 0.5
    W[8, 20] = -0.5
    
    # Term 21: (A01+A12)*(B10+B21)
    # A01*B10 (C00), A01*B21 (Mismatch), A12*B10 (Mismatch), A12*B21 (C11).
    U[1, 21] = 1; U[5, 21] = 1
    V[3, 21] = 1; V[7, 21] = 1
    W[0, 21] = 0.5
    W[4, 21] = 0.5
    
    # Term 22: (A01-A12)*(B10-B21)
    U[1, 22] = 1; U[5, 22] = -1
    V[3, 22] = 1; V[7, 22] = -1
    W[0, 22] = 0.5
    W[4, 22] = -0.5

    # Verify
    def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > tol:
                        return False
        return True

    if _verify(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Fallback Rank 27
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
