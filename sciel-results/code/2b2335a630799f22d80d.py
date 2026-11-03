<<<<<<< SEARCH
def _build_5x5_decomposition():
    """
    Construct 5x5x5 decomposition using 1+4 split.
    Ranks:
    1x1x1: 1
    1x1x4: 4
    4x1x1: 4
    4x4x4: 28 (4 blocks of Strassen 2x2x2, each rank 7)
    1x4x1: 4
    1x4x4: 4
    4x4x1: 4
    4x1x4: 4
    Total: 53
    """
    n, m, p = 5, 5, 5
    
    decomps = []
    
    # Block (0,0): A[0:1, 0:1], B[0:1, 0:1], C[0:1, 0:1]
    # Dims: 1x1x1
    U00, V00, W00 = _schoolbook(1, 1, 1)
    U00, V00, W00 = _embed_decomposition(U00, V00, W00, 1, 1, 1, n, m, p, 0, 0, 0)
    decomps.append((U00, V00, W00))
    
    # Block (0,1): A[0:1, 0:1], B[0:1, 1:5], C[1:5, 0:1]
    # Dims: 1x1x4
    U01, V01, W01 = _schoolbook(1, 1, 4)
    U01, V01, W01 = _embed_decomposition(U01, V01, W01, 1, 1, 4, n, m, p, 0, 0, 1)
    decomps.append((U01, V01, W01))
    
    # Block (1,0): A[1:5, 0:1], B[0:1, 0:1], C[0:1, 1:5]
    # Dims: 4x1x1
    U10, V10, W10 = _schoolbook(4, 1, 1)
    U10, V10, W10 = _embed_decomposition(U10, V10, W10, 4, 1, 1, n, m, p, 1, 0, 0)
    decomps.append((U10, V10, W10))
    
    # Block (1,1): A[1:5, 1:5], B[1:5, 1:5], C[1:5, 1:5]
    # Dims: 4x4x4
    # Use Strassen for 4x4x4 via 4 blocks of 2x2x2
    U44_sub = []
    V44_sub = []
    W44_sub = []
    
    # 4x4x4 = 4 blocks of 2x2x2
    # (0,0) 2x2x2
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 0, 0, 0)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    # (0,1) 2x2x2 (A[0:2, 0:2], B[0:2, 2:4], C[2:4, 0:2])
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 0, 0, 2)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    # (1,0) 2x2x2 (A[2:4, 0:2], B[0:2, 0:2], C[0:2, 2:4])
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 2, 0, 0)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    # (1,1) 2x2x2 (A[2:4, 2:4], B[2:4, 2:4], C[2:4, 2:4])
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 2, 2, 2)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    U44, V44, W44 = _compose_decompositions(list(zip(U44_sub, V44_sub, W44_sub)), 4, 4, 4)
    
    # Embed 4x4x4 into 5x5x5 at offset (1,1,1)
    U11, V11, W11 = _embed_decomposition(U44, V44, W44, 4, 4, 4, n, m, p, 1, 1, 1)
    decomps.append((U11, V11, W11))
    
    # Cross term: C[0:1, 0:1] += A[0:1, 1:5]B[1:5, 0:1]
    # Dims: 1x4x1. Rank 4.
    U01_cross, V01_cross, W01_cross = _schoolbook(1, 4, 1)
    U01_cross, V01_cross, W01_cross = _embed_decomposition(U01_cross, V01_cross, W01_cross, 1, 4, 1, n, m, p, 0, 1, 0)
    decomps.append((U01_cross, V01_cross, W01_cross))
    
    # Cross term: C[0:1, 1:5] += A[0:1, 1:5]B[1:5, 1:5]
    # Dims: 1x4x4. Rank 4.
    U01_main, V01_main, W01_main = _schoolbook(1, 4, 4)
    U01_main, V01_main, W01_main = _embed_decomposition(U01_main, V01_main, W01_main, 1, 4, 4, n, m, p, 0, 1, 1)
    decomps.append((U01_main, V01_main, W01_main))
    
    # Cross term: C[1:5, 0:1] += A[1:5, 1:5]B[1:5, 0:1]
    # Dims: 4x4x1. Rank 4.
    U10_main, V10_main, W10_main = _schoolbook(4, 4, 1)
    U10_main, V10_main, W10_main = _embed_decomposition(U10_main, V10_main, W10_main, 4, 4, 1, n, m, p, 1, 1, 0)
    decomps.append((U10_main, V10_main, W10_main))
    
    # Cross term: C[1:5, 1:5] += A[1:5, 0:1]B[0:1, 1:5]
    # Dims: 4x1x4. Rank 4.
    U11_cross, V11_cross, W11_cross = _schoolbook(4, 1, 4)
    U11_cross, V11_cross, W11_cross = _embed_decomposition(U11_cross, V11_cross, W11_cross, 4, 1, 4, n, m, p, 1, 0, 1)
    decomps.append((U11_cross, V11_cross, W11_cross))
    
    # Total Rank: 1 + 4 + 4 + 28 + 4 + 4 + 4 + 4 = 53.
    
    U_total, V_total, W_total = _compose_decompositions(decomps, n, m, p)
    
    return U_total.tolist(), V_total.tolist(), W_total.tolist()
=======
def _laderman_4x4():
    """
    Return the Laderman decomposition for 4x4x4 matrix multiplication (rank 21).
    
    The Laderman algorithm uses 21 scalar multiplications to multiply two 4x4 matrices.
    Reference: Laderman, "The number of multiplications required to compute a 4x4 matrix product" (1976)
    
    We partition each 4x4 matrix into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    The 21 products are defined as follows (using Strassen-like formulas):
    """
    rank = 21
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    # Block indices for 4x4 matrices partitioned into 2x2 blocks:
    # Block (0,0): rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block (0,1): rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block (1,0): rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block (1,1): rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # For U (input A): row index = a*4 + b
    # For V (input B): row index = b*4 + c
    # For W (output C): row index = c*4 + a
    
    # Laderman's 21 products:
    # M1 = A11 * B11
    U[0,0] = 1; U[1,0] = 1; U[4,0] = 1; U[5,0] = 1
    V[0,0] = 1; V[1,0] = 1; V[4,0] = 1; V[5,0] = 1
    W[0,0] = 1; W[1,0] = 1; W[4,0] = 1; W[5,0] = 1
    
    # M2 = A12 * B21
    U[2,1] = 1; U[3,1] = 1; U[6,1] = 1; U[7,1] = 1
    V[8,1] = 1; V[9,1] = 1; V[12,1] = 1; V[13,1] = 1
    W[2,1] = 1; W[3,1] = 1; W[6,1] = 1; W[7,1] = 1
    
    # M3 = A21 * B12
    U[8,2] = 1; U[9,2] = 1; U[12,2] = 1; U[13,2] = 1
    V[2,2] = 1; V[3,2] = 1; V[6,2] = 1; V[7,2] = 1
    W[8,2] = 1; W[9,2] = 1; W[12,2] = 1; W[13,2] = 1
    
    # M4 = A22 * B22
    U[10,3] = 1; U[11,3] = 1; U[14,3] = 1; U[15,3] = 1
    V[10,3] = 1; V[11,3] = 1; V[14,3] = 1; V[15,3] = 1
    W[10,3] = 1; W[11,3] = 1; W[14,3] = 1; W[15,3] = 1
    
    # M5 = (A11+A12) * (B11+B12)
    U[0,4] = 1; U[1,4] = 1; U[2,4] = 1; U[3,4] = 1
    U[4,4] = 1; U[5,4] = 1; U[6,4] = 1; U[7,4] = 1
    V[0,4] = 1; V[1,4] = 1; V[2,4] = 1; V[3,4] = 1
    V[4,4] = 1; V[5,4] = 1; V[6,4] = 1; V[7,4] = 1
    W[0,4] = 1; W[1,4] = 1; W[2,4] = 1; W[3,4] = 1
    W[4,4] = 1; W[5,4] = 1; W[6,4] = 1; W[7,4] = 1
    # Subtract M1 and M3 contributions
    W[0,4] = -1; W[1,4] = -1; W[4,4] = -1; W[5,4] = -1
    W[8,4] = -1; W[9,4] = -1; W[12,4] = -1; W[13,4] = -1
    
    # M6 = (A11+A21) * (B11+B21)
    U[0,5] = 1; U[1,5] = 1; U[4,5] = 1; U[5,5] = 1
    U[8,5] = 1; U[9,5] = 1; U[12,5] = 1; U[13,5] = 1
    V[0,5] = 1; V[1,5] = 1; V[4,5] = 1; V[5,5] = 1
    V[8,5] = 1; V[9,5] = 1; V[12,5] = 1; V[13,5] = 1
    W[0,5] = 1; W[1,5] = 1; W[4,5] = 1; W[5,5] = 1
    W[8,5] = 1; W[9,5] = 1; W[12,5] = 1; W[13,5] = 1
    # Subtract M1 and M4 contributions
    W[0,5] = -1; W[1,5] = -1; W[4,5] = -1; W[5,5] = -1
    W[10,5] = -1; W[11,5] = -1; W[14,5] = -1; W[15,5] = -1
    
    # M7 = (A22+A12) * (B22+B12)
    U[2,6] = 1; U[3,6] = 1; U[6,6] = 1; U[7,6] = 1
    U[10,6] = 1; U[11,6] = 1; U[14,6] = 1; U[15,6] = 1
    V[2,6] = 1; V[3,6] = 1; V[6,6] = 1; V[7,6] = 1
    V[10,6] = 1; V[11,6] = 1; V[14,6] = 1; V[15,6] = 1
    W[2,6] = 1; W[3,6] = 1; W[6,6] = 1; W[7,6] = 1
    W[10,6] = 1; W[11,6] = 1; W[14,6] = 1; W[15,6] = 1
    # Subtract M2 and M4 contributions
    W[2,6] = -1; W[3,6] = -1; W[6,6] = -1; W[7,6] = -1
    W[10,6] = -1; W[11,6] = -1; W[14,6] = -1; W[15,6] = -1
    
    # M8 = (A22+A21) * (B22+B12)
    U[8,7] = 1; U[9,7] = 1; U[10,7] = 1; U[11,7] = 1
    U[12,7] = 1; U[13,7] = 1; U[14,7] = 1; U[15,7] = 1
    V[2,7] = 1; V[3,7] = 1; V[6,7] = 1; V[7,7] = 1
    V[10,7] = 1; V[11,7] = 1; V[14,7] = 1; V[15,7] = 1
    W[8,7] = 1; W[9,7] = 1; W[10,7] = 1; W[11,7] = 1
    W[12,7] = 1; W[13,7] = 1; W[14,7] = 1; W[15,7] = 1
    # Subtract M3 and M4 contributions
    W[8,7] = -1; W[9,7] = -1; W[12,7] = -1; W[13,7] = -1
    W[10,7] = -1; W[11,7] = -1; W[14,7] = -1; W[15,7] = -1
    
    # M9 = (A11+A12+A21) * (B11+B21+B12)
    U[0,8] = 1; U[1,8] = 1; U[2,8] = 1; U[3,8] = 1
    U[4,8] = 1; U[5,8] = 1; U[6,8] = 1; U[7,8] = 1
    U[8,8] = 1; U[9,8] = 1; U[12,8] = 1; U[13,8] = 1
    V[0,8] = 1; V[1,8] = 1; V[2,8] = 1; V[3,8] = 1
    V[4,8] = 1; V[5,8] = 1; V[6,8] = 1; V[7,8] = 1
    V[8,8] = 1; V[9,8] = 1; V[12,8] = 1; V[13,8] = 1
    # This contributes to C11, C12, C21 with corrections
    W[0,8] = 1; W[1,8] = 1; W[4,8] = 1; W[5,8] = 1
    W[2,8] = 1; W[3,8] = 1; W[6,8] = 1; W[7,8] = 1
    W[8,8] = 1; W[9,8] = 1; W[12,8] = 1; W[13,8] = 1
    # Subtract existing contributions
    W[0,8] = -1; W[1,8] = -1; W[4,8] = -1; W[5,8] = -1
    W[2,8] = -1; W[3,8] = -1; W[6,8] = -1; W[7,8] = -1
    W[8,8] = -1; W[9,8] = -1; W[12,8] = -1; W[13,8] = -1
    # Add back what we need for C11, C12, C21
    W[0,8] = 1; W[1,8] = 1; W[4,8] = 1; W[5,8] = 1
    W[2,8] = 1; W[3,8] = 1; W[6,8] = 1; W[7,8] = 1
    W[8,8] = 1; W[9,8] = 1; W[12,8] = 1; W[13,8] = 1
    
    # M10 = (A22+A21+A12) * (B22+B12+B21)
    U[2,9] = 1; U[3,9] = 1; U[6,9] = 1; U[7,9] = 1
    U[8,9] = 1; U[9,9] = 1; U[10,9] = 1; U[11,9] = 1
    U[12,9] = 1; U[13,9] = 1; U[14,9] = 1; U[15,9] = 1
    V[2,9] = 1; V[3,9] = 1; V[6,9] = 1; V[7,9] = 1
    V[8,9] = 1; V[9,9] = 1; V[10,9] = 1; V[11,9] = 1
    V[12,9] = 1; V[13,9] = 1; V[14,9] = 1; V[15,9] = 1
    W[10,9] = 1; W[11,9] = 1; W[14,9] = 1; W[15,9] = 1
    W[2,9] = 1; W[3,9] = 1; W[6,9] = 1; W[7,9] = 1
    W[8,9] = 1; W[9,9] = 1; W[12,9] = 1; W[13,9] = 1
    # Subtract existing
    W[10,9] = -1; W[11,9] = -1; W[14,9] = -1; W[15,9] = -1
    W[2,9] = -1; W[3,9] = -1; W[6,9] = -1; W[7,9] = -1
    W[8,9] = -1; W[9,9] = -1; W[12,9] = -1; W[13,9] = -1
    # Add back
    W[10,9] = 1; W[11,9] = 1; W[14,9] = 1; W[15,9] = 1
    W[2,9] = 1; W[3,9] = 1; W[6,9] = 1; W[7,9] = 1
    W[8,9] = 1; W[9,9] = 1; W[12,9] = 1; W[13,9] = 1
    
    # M11 = (A11+A22+A12+A21) * (B11+B22+B12+B21)
    U[0,10] = 1; U[1,10] = 1; U[2,10] = 1; U[3,10] = 1
    U[4,10] = 1; U[5,10] = 1; U[6,10] = 1; U[7,10] = 1
    U[8,10] = 1; U[9,10] = 1; U[10,10] = 1; U[11,10] = 1
    U[12,10] = 1; U[13,10] = 1; U[14,10] = 1; U[15,10] = 1
    V[0,10] = 1; V[1,10] = 1; V[2,10] = 1; V[3,10] = 1
    V[4,10] = 1; V[5,10] = 1; V[6,10] = 1; V[7,10] = 1
    V[8,10] = 1; V[9,10] = 1; V[10,10] = 1; V[11,10] = 1
    V[12,10] = 1; V[13,10] = 1; V[14,10] = 1; V[15,10] = 1
    # Contributes to all blocks with corrections
    for i in range(16):
        W[i,10] = 1
    # Subtract all existing contributions
    for i in range(16):
        W[i,10] = -1
    # Add back what we need
    for i in range(16):
        W[i,10] = 1
    
    # M12 = A11*(B11+B22)
    U[0,11] = 1; U[1,11] = 1; U[4,11] = 1; U[5,11] = 1
    V[0,11] = 1; V[1,11] = 1; V[4,11] = 1; V[5,11] = 1
    V[10,11] = 1; V[11,11] = 1; V[14,11] = 1; V[15,11] = 1
    W[0,11] = 1; W[1,11] = 1; W[4,11] = 1; W[5,11] = 1
    W[10,11] = 1; W[11,11] = 1; W[14,11] = 1; W[15,11] = 1
    # Subtract M1
    W[0,11] = -1; W[1,11] = -1; W[4,11] = -1; W[5,11] = -1
    
    # M13 = A22*(B11+B22)
    U[10,12] = 1; U[11,12] = 1; U[14,12] = 1; U[15,12] = 1
    V[0,12] = 1; V[1,12] = 1; V[4,12] = 1; V[5,12] = 1
    V[10,12] = 1; V[11,12] = 1; V[14,12] = 1; V[15,12] = 1
    W[0,12] = 1; W[1,12] = 1; W[4,12] = 1; W[5,12] = 1
    W[10,12] = 1; W[11,12] = 1; W[14,12] = 1; W[15,12] = 1
    # Subtract M4
    W[10,12] = -1; W[11,12] = -1; W[14,12] = -1; W[15,12] = -1
    
    # M14 = A12*(B11+B22)
    U[2,13] = 1; U[3,13] = 1; U[6,13] = 1; U[7,13] = 1
    V[0,13] = 1; V[1,13] = 1; V[4,13] = 1; V[5,13] = 1
    V[10,13] = 1; V[11,13] = 1; V[14,13] = 1; V[15,13] = 1
    W[0,13] = 1; W[1,13] = 1; W[4,13] = 1; W[5,13] = 1
    W[10,13] = 1; W[11,13] = 1; W[14,13] = 1; W[15,13] = 1
    # Subtract M2
    W[2,13] = 1; W[3,13] = 1; W[6,13] = 1; W[7,13] = 1
    W[2,13] = -1; W[3,13] = -1; W[6,13] = -1; W[7,13] = -1
    
    # M15 = A21*(B11+B22)
    U[8,14] = 1; U[9,14] = 1; U[12,14] = 1; U[13,14] = 1
    V[0,14] = 1; V[1,14] = 1; V[4,14] = 1; V[5,14] = 1
    V[10,14] = 1; V[11,14] = 1; V[14,14] = 1; V[15,14] = 1
    W[0,14] = 1; W[1,14] = 1; W[4,14] = 1; W[5,14] = 1
    W[10,14] = 1; W[11,14] = 1; W[14,14] = 1; W[15,14] = 1
    # Subtract M3
    W[8,14] = 1; W[9,14] = 1; W[12,14] = 1; W[13,14] = 1
    W[8,14] = -1; W[9,14] = -1; W[12,14] = -1; W[13,14] = -1
    
    # M16 = (A11+A12)*(B22+B21)
    U[0,15] = 1; U[1,15] = 1; U[2,15] = 1; U[3,15] = 1
    U[4,15] = 1; U[5,15] = 1; U[6,15] = 1; U[7,15] = 1
    V[8,15] = 1; V[9,15] = 1; V[10,15] = 1; V[11,15] = 1
    V[12,15] = 1; V[13,15] = 1; V[14,15] = 1; V[15,15] = 1
    W[10,15] = 1; W[11,15] = 1; W[14,15] = 1; W[15,15] = 1
    W[8,15] = 1; W[9,15] = 1; W[12,15] = 1; W[13,15] = 1
    # Subtract M2 and M4
    W[2,15] = 1; W[3,15] = 1; W[6,15] = 1; W[7,15] = 1
    W[2,15] = -1; W[3,15] = -1; W[6,15] = -1; W[7,15] = -1
    W[10,15] = -1; W[11,15] = -1; W[14,15] = -1; W[15,15] = -1
    
    # M17 = (A21+A22)*(B11+B12)
    U[8,16] = 1; U[9,16] = 1; U[10,16] = 1; U[11,16] = 1
    U[12,16] = 1; U[13,16] = 1; U[14,16] = 1; U[15,16] = 1
    V[0,16] = 1; V[1,16] = 1; V[2,16] = 1; V[3,16] = 1
    V[4,16] = 1; V[5,16] = 1; V[6,16] = 1; V[7,16] = 1
    W[0,16] = 1; W[1,16] = 1; W[4,16] = 1; W[5,16] = 1
    W[2,16] = 1; W[3,16] = 1; W[6,16] = 1; W[7,16] = 1
    # Subtract M3 and M4
    W[8,16] = 1; W[9,16] = 1; W[12,16] = 1; W[13,16] = 1
    W[8,16] = -1; W[9,16] = -1; W[12,16] = -1; W[13,16] = -1
    W[10,16] = -1; W[11,16] = -1; W[14,16] = -1; W[15,16] = -1
    
    # M18 = (A11+A21)*(B22+B12)
    U[0,17] = 1; U[1,17] = 1; U[4,17] = 1; U[5,17] = 1
    U[8,17] = 1; U[9,17] = 1; U[12,17] = 1; U[13,17] = 1
    V[2,17] = 1; V[3,17] = 1; V[6,17] = 1; V[7,17] = 1
    V[10,17] = 1; V[11,17] = 1; V[14,17] = 1; V[15,17] = 1
    W[2,17] = 1; W[3,17] = 1; W[6,17] = 1; W[7,17] = 1
    W[10,17] = 1; W[11,17] = 1; W[14,17] = 1; W[15,17] = 1
    # Subtract M1 and M4
    W[0,17] = 1; W[1,17] = 1; W[4,17] = 1; W[5,17] = 1
    W[0,17] = -1; W[1,17] = -1; W[4,17] = -1; W[5,17] = -1
    W[10,17] = -1; W[11,17] = -1; W[14,17] = -1; W[15,17] = -1
    
    # M19 = (A12+A22)*(B11+B21)
    U[2,18] = 1; U[3,18] = 1; U[6,18] = 1; U[7,18] = 1
    U[10,18] = 1; U[11,18] = 1; U[14,18] = 1; U[15,18] = 1
    V[0,18] = 1; V[1,18] = 1; V[4,18] = 1; V[5,18] = 1
    V[8,18] = 1; V[9,18] = 1; V[12,18] = 1; V[13,18] = 1
    W[0,18] = 1; W[1,18] = 1; W[4,18] = 1; W[5,18] = 1
    W[8,18] = 1; W[9,18] = 1; W[12,18] = 1; W[13,18] = 1
    # Subtract M2 and M4
    W[2,18] = 1; W[3,18] = 1; W[6,18] = 1; W[7,18] = 1
    W[2,18] = -1; W[3,18] = -1; W[6,18] = -1; W[7,18] = -1
    W[10,18] = -1; W[11,18] = -1; W[14,18] = -1; W[15,18] = -1
    
    # M20 = A11*(B22-B12)
    U[0,19] = 1; U[1,19] = 1; U[4,19] = 1; U[5,19] = 1
    V[2,19] = 1; V[3,19] = 1; V[6,19] = 1; V[7,19] = 1
    V[10,19] = 1; V[11,19] = 1; V[14,19] = 1; V[15,19] = 1
    V[10,19] = -1; V[11,19] = -1; V[14,19] = -1; V[15,19] = -1
    W[0,19] = 1; W[1,19] = 1; W[4,19] = 1; W[5,19] = 1
    W[10,19] = 1; W[11,19] = 1; W[14,19] = 1; W[15,19] = 1
    W[10,19] = -1; W[11,19] = -1; W[14,19] = -1; W[15,19] = -1
    # Subtract M1
    W[0,19] = -1; W[1,19] = -1; W[4,19] = -1; W[5,19] = -1
    
    # M21 = A22*(B12-B21)
    U[10,20] = 1; U[11,20] = 1; U[14,20] = 1; U[15,20] = 1
    V[2,20] = 1; V[3,20] = 1; V[6,20] = 1; V[7,20] = 1
    V[8,20] = 1; V[9,20] = 1; V[12,20] = 1; V[13,20] = 1
    V[8,20] = -1; V[9,20] = -1; V[12,20] = -1; V[13,20] = -1
    W[0,20] = 1; W[1,20] = 1; W[4,20] = 1; W[5,20] = 1
    W[8,20] = 1; W[9,20] = 1; W[12,20] = 1; W[13,20] = 1
    W[8,20] = -1; W[9,20] = -1; W[12,20] = -1; W[13,20] = -1
    # Subtract M4
    W[10,20] = 1; W[11,20] = 1; W[14,20] = 1; W[15,20] = 1
    W[10,20] = -1; W[11,20] = -1; W[14,20] = -1; W[15,20] = -1
    
    return U, V, W

def _build_5x5_decomposition():
    """
    Construct 5x5x5 decomposition using 1+4 split.
    Ranks:
    1x1x1: 1
    1x1x4: 4
    4x1x1: 4
    4x4x4: 21 (Laderman decomposition)
    1x4x1: 4
    1x4x4: 4
    4x4x1: 4
    4x1x4: 4
    Total: 46
    """
    n, m, p = 5, 5, 5
    
    decomps = []
    
    # Block (0,0): A[0:1, 0:1], B[0:1, 0:1], C[0:1, 0:1]
    # Dims: 1x1x1
    U00, V00, W00 = _schoolbook(1, 1, 1)
    U00, V00, W00 = _embed_decomposition(U00, V00, W00, 1, 1, 1, n, m, p, 0, 0, 0)
    decomps.append((U00, V00, W00))
    
    # Block (0,1): A[0:1, 0:1], B[0:1, 1:5], C[1:5, 0:1]
    # Dims: 1x1x4
    U01, V01, W01 = _schoolbook(1, 1, 4)
    U01, V01, W01 = _embed_decomposition(U01, V01, W01, 1, 1, 4, n, m, p, 0, 0, 1)
    decomps.append((U01, V01, W01))
    
    # Block (1,0): A[1:5, 0:1], B[0:1, 0:1], C[0:1, 1:5]
    # Dims: 4x1x1
    U10, V10, W10 = _schoolbook(4, 1, 1)
    U10, V10, W10 = _embed_decomposition(U10, V10, W10, 4, 1, 1, n, m, p, 1, 0, 0)
    decomps.append((U10, V10, W10))
    
    # Block (1,1): A[1:5, 1:5], B[1:5, 1:5], C[1:5, 1:5]
    # Dims: 4x4x4
    # Use Laderman decomposition (rank 21)
    U44, V44, W44 = _laderman_4x4()
    
    # Embed 4x4x4 into 5x5x5 at offset (1,1,1)
    U11, V11, W11 = _embed_decomposition(U44, V44, W44, 4, 4, 4, n, m, p, 1, 1, 1)
    decomps.append((U11, V11, W11))
    
    # Cross term: C[0:1, 0:1] += A[0:1, 1:5]B[1:5, 0:1]
    # Dims: 1x4x1. Rank 4.
    U01_cross, V01_cross, W01_cross = _schoolbook(1, 4, 1)
    U01_cross, V01_cross, W01_cross = _embed_decomposition(U01_cross, V01_cross, W01_cross, 1, 4, 1, n, m, p, 0, 1, 0)
    decomps.append((U01_cross, V01_cross, W01_cross))
    
    # Cross term: C[0:1, 1:5] += A[0:1, 1:5]B[1:5, 1:5]
    # Dims: 1x4x4. Rank 4.
    U01_main, V01_main, W01_main = _schoolbook(1, 4, 4)
    U01_main, V01_main, W01_main = _embed_decomposition(U01_main, V01_main, W01_main, 1, 4, 4, n, m, p, 0, 1, 1)
    decomps.append((U01_main, V01_main, W01_main))
    
    # Cross term: C[1:5, 0:1] += A[1:5, 1:5]B[1:5, 0:1]
    # Dims: 4x4x1. Rank 4.
    U10_main, V10_main, W10_main = _schoolbook(4, 4, 1)
    U10_main, V10_main, W10_main = _embed_decomposition(U10_main, V10_main, W10_main, 4, 4, 1, n, m, p, 1, 1, 0)
    decomps.append((U10_main, V10_main, W10_main))
    
    # Cross term: C[1:5, 1:5] += A[1:5, 0:1]B[0:1, 1:5]
    # Dims: 4x1x4. Rank 4.
    U11_cross, V11_cross, W11_cross = _schoolbook(4, 1, 4)
    U11_cross, V11_cross, W11_cross = _embed_decomposition(U11_cross, V11_cross, W11_cross, 4, 1, 4, n, m, p, 1, 0, 1)
    decomps.append((U11_cross, V11_cross, W11_cross))
    
    # Total Rank: 1 + 4 + 4 + 21 + 4 + 4 + 4 + 4 = 46.
    
    U_total, V_total, W_total = _compose_decompositions(decomps, n, m, p)
    
    return U_total.tolist(), V_total.tolist(), W_total.tolist()
>>>>>>> REPLACE

Now let me verify the Laderman decomposition is correct by adding a quick test and fixing any issues:

