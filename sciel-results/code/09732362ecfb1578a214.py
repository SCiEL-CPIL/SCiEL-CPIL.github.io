import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # 2x3 matrix A, 3x3 matrix B, 2x3 matrix C
    # A indices: i*m+j = i*3+j -> A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B indices: j*p+k = j*3+k -> B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # C indices: k*n+i = k*2+i -> C00=0, C10=1, C01=2, C11=3, C02=4, C12=5
    
    # Rank 17 decomposition
    rank = 17
    U = np.zeros((n * m, rank))  # (6, 17)
    V = np.zeros((m * p, rank))  # (9, 17)
    W = np.zeros((p * n, rank))  # (6, 17)
    
    # We'll define each rank-1 term as a tuple of (U_indices, V_indices, W_indices, values)
    # Actually, let's just build it directly.
    
    # Block 1: 2x2x2 Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A[0:2,0:2]: A00(0), A01(1), A10(3), A11(4)
    # B[0:2,0:2]: B00(0), B01(1), B10(3), B11(4)
    # C[0:2,0:2]: C00(0), C01(2), C10(1), C11(3)
    
    # r=0: M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00 -> contributes to C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11) -> contributes to C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00) -> contributes to C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11 -> contributes to -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01) -> contributes to C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11) -> contributes to C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Block 2: A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    # A: A02(2), A12(5)
    # B: B20(6), B21(7)
    # C: C00(0), C01(2), C10(1), C11(3)
    
    # r=7: A02 * B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 * B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 * B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 * B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Block 3: A[0:2,0:2] * B[0:2,2] -> C[0:2,2]
    # A: A00(0), A01(1), A10(3), A11(4)
    # B: B02(2), B12(5)
    # C: C02(4), C12(5)
    
    # r=11: A00 * B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 * B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A10 * B02 -> C12
    U[3, 13] = 1
    V[2, 13] = 1
    W[5, 13] = 1
    
    # r=14: A11 * B12 -> C12
    U[4, 14] = 1
    V[5, 14] = 1
    W[5, 14] = 1
    
    # Block 4: A[0:2,2] * B[2,2] -> C[0:2,2]
    # A: A02(2), A12(5)
    # B: B22(8)
    # C: C02(4), C12(5)
    
    # r=15: A02 * B22 -> C02
    U[2, 15] = 1
    V[8, 15] = 1
    W[4, 15] = 1
    
    # r=16: A12 * B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
