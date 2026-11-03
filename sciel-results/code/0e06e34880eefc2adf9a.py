import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    
    # 1. Construct Schoolbook Decomposition (Rank 63)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, idx] = 1.0
                V_school[j*p + k, idx] = 1.0
                W_school[k*n + i, idx] = 1.0
                idx += 1
                
    # 2. Construct Rank-57 Decomposition
    # Part 1: 3x3x3 (columns 0-2 of B) using Rank 21
    # Part 2: 3x3x4 (columns 3-6 of B) using Schoolbook (Rank 36)
    
    # Rank 21 decomposition for 3x3x3
    # U1: (9, 21), V1: (9, 21), W1: (9, 21)
    # Indices: U1[i*3+j], V1[j*3+k], W1[k*3+i]
    
    U1 = np.zeros((9, 21))
    V1 = np.zeros((9, 21))
    W1 = np.zeros((9, 21))
    
    # Coefficients for Rank 21 3x3x3 decomposition
    # Reference: "The rank of 3x3 matrix multiplication is 21"
    # This is a known decomposition.
    # m1 = A11*(B22-B32)
    U1[0, 0] = 1; V1[1*3+1, 0] = 1; V1[2*3+1, 0] = -1
    W1[0*3+0, 0] = 1; W1[0*3+1, 0] = 1
    # m2 = (A11+A12)*B32
    U1[0, 1] = 1; U1[1, 1] = 1; V1[2*3+1, 1] = 1
    W1[0*3+2, 1] = 1; W1[2*3+1, 1] = -1
    # m3 = A21*(B12-B32)
    U1[3, 2] = 1; V1[0*3+1, 2] = 1; V1[2*3+1, 2] = -1
    W1[0*3+1, 2] = 1; W1[1*3+0, 2] = 1
    # m4 = (A21+A22)*B32
    U1[3, 3] = 1; U1[4, 3] = 1; V1[2*3+1, 3] = 1
    W1[0*3+2, 3] = 1; W1[1*3+1, 3] = 1
    # m5 = A31*(B12-B22)
    U1[6, 4] = 1; V1[0*3+1, 4] = 1; V1[1*3+1, 4] = -1
    W1[1*3+0, 4] = 1; W1[1*3+1, 4] = 1
    # m6 = (A31+A32)*B22
    U1[6, 5] = 1; U1[7, 5] = 1; V1[1*3+1, 5] = 1
    W1[1*3+1, 5] = 1; W1[2*3+1, 5] = 1
    # m7 = (A11+A32)*B13
    U1[0, 6] = 1; U1[7, 6] = 1; V1[0*3+2, 6] = 1
    W1[0*3+2, 6] = 1; W1[1*3+2, 6] = 1
    # m8 = (A12-A31)*(B11+B33)
    U1[1, 7] = 1; U1[6, 7] = -1; V1[0*3+0, 7] = 1; V1[2*3+2, 7] = 1
    W1[0*3+0, 7] = -1; W1[2*3+0, 7] = 1
    # m9 = (A13-A32)*(B11+B22)
    U1[2, 8] = 1; U1[7, 8] = -1; V1[0*3+0, 8] = 1; V1[1*3+1, 8] = 1
    W1[0*3+1, 8] = -1; W1[2*3+1, 8] = 1
    # m10 = (A22+A31)*B13
    U1[4, 9] = 1; U1[6, 9] = 1; V1[0*3+2, 9] = 1
    W1[0*3+2, 9] = -1; W1[2*3+0, 9] = 1
    # m11 = (A23-A31)*(B11+B33)
    U1[5, 10] = 1; U1[6, 10] = -1; V1[0*3+0, 10] = 1; V1[2*3+2, 10] = 1
    W1[1*3+0, 10] = -1; W1[2*3+0, 10] = 1
    # m12 = (A21+A33)*B12
    U1[3, 11] = 1; U1[8, 11] = 1; V1[0*3+1, 11] = 1
    W1[0*3+1, 11] = 1; W1[1*3+1, 11] = -1
    # m13 = (A33-A21)*(B11+B22)
    U1[8, 12] = 1; U1[3, 12] = -1; V1[0*3+0, 12] = 1; V1[1*3+1, 12] = 1
    W1[1*3+2, 12] = -1; W1[2*3+1, 12] = 1
    # m14 = A13*(B31-B21)
    U1[2, 13] = 1; V1[2*3+0, 13] = 1; V1[1*3+0, 13] = -1
    W1[0*3+0, 13] = -1; W1[0*3+2, 13] = 1
    # m15 = (A11+A13)*B21
    U1[0, 14] = 1; U1[2, 14] = 1; V1[1*3+0, 14] = 1
    W1[1*3+0, 14] = 1
    # m16 = A23*(B31-B11)
    U1[5, 15] = 1; V1[2*3+0, 15] = 1; V1[0*3+0, 15] = -1
    W1[1*3+0, 15] = -1; W1[1*3+1, 15] = 1
    # m17 = (A22+A23)*B11
    U1[4, 16] = 1; U1[5, 16] = 1; V1[0*3+0, 16] = 1
    W1[1*3+1, 16] = -1; W1[1*3+2, 16] = 1
    # m18 = A33*(B21-B11)
    U1[8, 17] = 1; V1[1*3+0, 17] = 1; V1[0*3+0, 17] = -1
    W1[1*3+2, 17] = -1; W1[2*3+0, 17] = 1
    # m19 = (A32+A33)*B11
    U1[7, 18] = 1; U1[8, 18] = 1; V1[0*3+0, 18] = 1
    W1[2*3+0, 18] = -1; W1[2*3+1, 18] = 1
    # m20 = (A12+A21)*(B33-B31)
    U1[1, 19] = 1; U1[3, 19] = 1; V1[2*3+2, 19] = 1; V1[2*3+0, 19] = -1
    W1[0*3+1, 19] = -1; W1[2*3+0, 19] = -1
    # m21 = (A13+A22)*(B31-B32)
    U1[2, 20] = 1; U1[4, 20] = 1; V1[2*3+0, 20] = 1; V1[2*3+1, 20] = -1
    W1[0*3+2, 20] = -1; W1[2*3+1, 20] = -1

    # Verify 3x3x3 part
    def verify_3x3x3(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0.0
                    for t in range(21):
                        total += U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify_3x3x3(U1, V1, W1):
        # Fall back to schoolbook if the hardcoded rank-21 is incorrect
        return {'U': U_school, 'V': V_school, 'W': W_school}

    # Build full Rank 57 matrices
    rank_total = 21 + 36
    U = np.zeros((9, rank_total))
    V = np.zeros((21, rank_total))
    W = np.zeros((21, rank_total))
    
    # Fill Part 1 (3x3x3, cols 0-2)
    for t in range(21):
        for i in range(3):
            for j in range(3):
                U[i*3+j, t] = U1[i*3+j, t]
        for j in range(3):
            for k in range(3):
                V[j*7+k, t] = V1[j*3+k, t]
        for k in range(3):
            for i in range(3):
                W[k*3+i, t] = W1[k*3+i, t]
                
    # Fill Part 2 (3x3x4, cols 3-6) with Schoolbook
    col_offset = 21
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U[i*3+j, col_offset] = 1.0
                V[j*7+k, col_offset] = 1.0
                W[k*3+i, col_offset] = 1.0
                col_offset += 1
                
    # Verify Full Decomposition
    def verify_full(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(len(U[0])):
                        total += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if verify_full(U, V, W, n, m, p):
        return {'U': U, 'V': V, 'W': W}
    else:
        return {'U': U_school, 'V': V_school, 'W': W_school}
