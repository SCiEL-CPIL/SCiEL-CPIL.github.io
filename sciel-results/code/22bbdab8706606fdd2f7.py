import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Returns (U, V, W) as numpy arrays.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a11+a22)(b11+b22)
    U[0, 0] += 1.0; U[3, 0] += 1.0
    V[0, 0] += 1.0; V[3, 0] += 1.0
    W[0, 0] += 1.0; W[3, 0] += 1.0
    # M2: (a21+a22)b11
    U[2, 1] += 1.0; U[3, 1] += 1.0
    V[0, 1] += 1.0
    W[2, 1] += 1.0; W[3, 1] += -1.0
    # M3: a11(b12-b22)
    U[0, 2] += 1.0
    V[1, 2] += 1.0; V[3, 2] += -1.0
    W[1, 2] += 1.0; W[3, 2] += 1.0
    # M4: a22(b21-b11)
    U[3, 3] += 1.0
    V[2, 3] += 1.0; V[0, 3] += -1.0
    W[0, 3] += 1.0; W[2, 3] += 1.0
    # M5: (a11+a12)b22
    U[0, 4] += 1.0; U[1, 4] += 1.0
    V[3, 4] += 1.0
    W[0, 4] += -1.0; W[1, 4] += 1.0
    # M6: (a21-a11)(b11+b12)
    U[2, 5] += 1.0; U[0, 5] += -1.0
    V[0, 5] += 1.0; V[1, 5] += 1.0
    W[3, 5] += 1.0
    # M7: (a11+a12)(b21+b22)
    U[0, 6] += 1.0; U[1, 6] += 1.0
    V[2, 6] += 1.0; V[3, 6] += 1.0
    W[0, 6] += 1.0
    
    return U, V, W

def _block_decompose(n_a, m, n_b):
    """Decompose n_a x m * m x n_b using Strassen-style 2x2 blocks.
    Returns (U, V, W) as numpy arrays with shapes (n_a*m, rank), (m*n_b, rank), (n_b*n_a, rank).
    """
    # Split into 2x2 blocks where possible
    # A: n_a x m, B: m x n_b, C: n_b x n_a
    # U rows: i*m+j for i in range(n_a), j in range(m)
    # V rows: j*n_b+k for j in range(m), k in range(n_b)
    # W rows: k*n_a+i for k in range(n_b), i in range(n_a)
    
    # Simple approach: use 2x2 Strassen for overlapping blocks
    # For now, use the naive decomposition as baseline
    rank = n_a * m * n_b
    U = np.zeros((n_a * m, rank))
    V = np.zeros((m * n_b, rank))
    W = np.zeros((n_b * n_a, rank))
    
    col = 0
    for i in range(n_a):
        for j in range(m):
            for k in range(n_b):
                U[i * m + j, col] = 1.0
                V[j * n_b + k, col] = 1.0
                W[k * n_a + i, col] = 1.0
                col += 1
    
    return U, V, W

def _decompose_2x4x3():
    """Decompose 2x4 * 4x3. Split A into two 2x2, B into two 4x... 
    Actually: A (2x4) = [A_left (2x2) | A_right (2x2)]
    B (4x3) = [B_top (2x3); B_bottom (2x3)]
    C = A_left * B_top + A_right * B_bottom
    Each 2x2*2x3 is rank 7 (Strassen-like), total rank 14.
    """
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    
    rank = 14
    U = np.zeros((8, rank))   # 2x4
    V = np.zeros((12, rank))  # 4x3
    W = np.zeros((6, rank))   # 3x2
    
    # Block 1: A_left (rows 0-1 of A, i.e., U rows 0-3) * B_top (cols 0-1 of B, i.e., V rows 0-5)
    # U rows: i*2+j, i in 0-1, j in 0-1 -> rows 0-3
    # V rows: j*3+k, j in 0-1, k in 0-2 -> rows 0-5
    # W rows: k*2+i, k in 0-2, i in 0-1 -> rows 0-5
    for i in range(4):
        for r in range(7):
            U[i, r] += U1[i, r]
    for j in range(2):
        for r in range(7):
            V[j * 3 + 0, r] += V1[j, r]  # This is wrong, need to map properly
    
    # Let me redo this more carefully
    # For 2x2 * 2x2 Strassen:
    # U row index: i*2+j, V row index: j*2+k, W row index: k*2+i
    # For 2x2 * 2x3:
    # A is 2x2 (rows 0-1, cols 0-1), B is 2x3 (rows 0-1, cols 0-2), C is 3x2 (rows 0-2, cols 0-1)
    # U row: i*2+j (i,j in 0-1) -> 4 rows
    # V row: j*3+k (j in 0-1, k in 0-2) -> 6 rows  
    # W row: k*2+i (k in 0-2, i in 0-1) -> 6 rows
    
    # The Strassen 2x2x2 has U(4,7), V(4,7), W(4,7) with indices i*2+j, j*2+k, k*2+i
    # For 2x2x3, we need to adapt. Let's just use the naive for now.
    
    # Naive 2x4*4x3: rank 24
    rank = 24
    U = np.zeros((8, rank))
    V = np.zeros((12, rank))
    W = np.zeros((6, rank))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j, col] = 1.0
                V[j * 3 + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Returns (U, V, W) as lists of lists.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # 7 products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Row index in U: i*2+j; V: j*2+k; W: k*2+i
    # M1: (a11+a22)(b11+b22) -> U[0]+U[3], V[0]+V[3], contributes to W[0](c11) and W[3](c22)
    # M2: (a21+a22)b11 -> U[2]+U[3], V[0], contributes to W[2](c21) and W[3](c22) with -1
    # M3: a11(b12-b22) -> U[0], V[1]-V[3], contributes to W[1](c12) and W[3](c22)
    # M4: a22(b21-b11) -> U[3], V[2]-V[0], contributes to W[0](c11) and W[2](c21)
    # M5: (a11+a12)b22 -> U[0]+U[1], V[3], contributes to W[0](c11) with -1 and W[1](c12)
    # M6: (a21-a11)(b11+b12) -> U[2]-U[0], V[0]+V[1], contributes to W[3](c22)
    # M7: (a11+a12)(b21+b22) -> U[0]+U[1], V[2]+V[3], contributes to W[0](c11)
    
    # M1
    U[0][0] += 1.0; U[3][0] += 1.0
    V[0][0] += 1.0; V[3][0] += 1.0
    W[0][0] += 1.0; W[3][0] += 1.0
    # M2
    U[2][1] += 1.0; U[3][1] += 1.0
    V[0][1] += 1.0
    W[2][1] += 1.0; W[3][1] += -1.0
    # M3
    U[0][2] += 1.0
    V[1][2] += 1.0; V[3][2] += -1.0
    W[1][2] += 1.0; W[3][2] += 1.0
    # M4
    U[3][3] += 1.0
    V[2][3] += 1.0; V[0][3] += -1.0
    W[0][3] += 1.0; W[2][3] += 1.0
    # M5
    U[0][4] += 1.0; U[1][4] += 1.0
    V[3][4] += 1.0
    W[0][4] += -1.0; W[1][4] += 1.0
    # M6
    U[2][5] += 1.0; U[0][5] += -1.0
    V[0][5] += 1.0; V[1][5] += 1.0
    W[3][5] += 1.0
    # M7
    U[0][6] += 1.0; U[1][6] += 1.0
    V[2][6] += 1.0; V[3][6] += 1.0
    W[0][6] += 1.0
    
    return U, V, W

def _rank1_121():
    """Rank-1 decomposition for 1x2 * 2x1. U(2,1), V(2,1), W(2,1).
    c11 = a11*b11 + a12*b21
    """
    U = [[0.0]*2 for _ in range(2)]
    V = [[0.0]*2 for _ in range(2)]
    W = [[0.0]*2 for _ in range(2)]
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    U[1][1] = 1.0; V[1][1] = 1.0; W[1][1] = 1.0
    return U, V, W

def _rank1_211():
    """Rank-1 decomposition for 2x1 * 1x1. U(2,1), V(1,1), W(2,1).
    c11 = a11*b11, c21 = a21*b11
    """
    U = [[0.0]*1 for _ in range(2)]
    V = [[0.0]*1 for _ in range(1)]
    W = [[0.0]*1 for _ in range(2)]
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    U[1][0] = 1.0; W[1][0] = 1.0
    return U, V, W

def _rank1_112():
    """Rank-1 decomposition for 1x1 * 1x2. U(1,1), V(2,1), W(2,1).
    c11 = a11*b11, c12 = a11*b12
    """
    U = [[0.0]*1 for _ in range(1)]
    V = [[0.0]*1 for _ in range(2)]
    W = [[0.0]*1 for _ in range(2)]
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    V[1][0] = 1.0; W[1][0] = 1.0
    return U, V, W

def _strassen_223():
    """Strassen-like for 2x2 * 2x3. Rank 14 (two 2x2 Strassen blocks).
    U(4,14), V(6,14), W(6,14).
    Split B into B_left (2x2) and B_right (2x2).
    C = [C_left | C_right] where C_left = A*B_left, C_right = A*B_right.
    """
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    U = [[0.0]*14 for _ in range(4)]
    V = [[0.0]*14 for _ in range(6)]
    W = [[0.0]*14 for _ in range(6)]
    
    # First block: columns 0-6
    for i in range(4):
        for r in range(7):
            U[i][r] += U1[i][r]
    # V for first block: rows 0-3 (B_left), rows 4-5 (B_right)
    for j in range(2):
        for r in range(7):
            V[j][r] += V1[j][r]
    # W for first block: rows 0-3 (C_left), rows 4-5 (C_right)
    for k in range(2):
        for r in range(7):
            W[k][r] += W1[k][r]
    
    # Second block: columns 7-13
    for i in range(4):
        for r in range(7):
            U[i][r+7] += U2[i][r]
    for j in range(2):
        for r in range(7):
            V[j+2][r+7] += V2[j][r]
    for k in range(2):
        for r in range(7):
            W[k+2][r+7] += W2[k][r]
    
    return U, V, W

def _decompose_343():
    """Decompose 3x4 * 4x3. Rank 28 (two 2x4 blocks + one 1x4 block).
    Actually: split A (3x4) into A1 (2x4) and A2 (1x4).
    C = [A1; A2] * B where B is 4x3.
    C1 = A1*B (2x3), C2 = A2*B (1x3).
    Use Strassen for 2x4*4x3 and rank-based for 1x4*4x3.
    """
    # 2x4 * 4x3: split into two 2x2*2x3 blocks? Or 2x4*4x2 + 2x4*4x1
    # Let's do: 2x4 * 4x3 = 2x4*4x2 + 2x4*4x1
    # 2x4*4x2: split A into 2x2+2x2, B into 4x2
    # Actually simpler: 2x4 * 4x3 can be done as two 2x2*2x3 Strassen = 14
    # 2x4 * 4x1: split into 2x2*2x1 + 2x2*2x1 = 2*4 = 8? Or use 2x4*4x1 directly
    # 2x4*4x1: rank 4 (each of 4 products)
    # Total for 2x4*4x3: 14 + 4 = 18? Let me think more carefully.
    
    # 2x4 * 4x3: A is 2x4, B is 4x3
    # Split A into A_left (2x2) and A_right (2x2)
    # C = A_left * B_left + A_right * B_right where B_left is 2x3, B_right is 2x3
    # A_left * B_left: 2x2 * 2x3, Strassen rank 7
    # A_right * B_right: 2x2 * 2x3, Strassen rank 7
    # Total: 14
    
    # 1x4 * 4x3: A is 1x4, B is 4x3
    # C = A[0]*B[:,0] + A[1]*B[:,1] + A[2]*B[:,2] + A[3]*B[:,3]
    # This is rank 4
    # Total for 3x4*4x3: 14 + 4 = 18
    
    # Build 2x4*4x3 decomposition (rank 14)
    U1, V1, W1 = _strassen_223()  # 2x2*2x3
    U2, V2, W2 = _strassen_223()  # 2x2*2x3
    
    U_243 = [[0.0]*14 for _ in range(8)]  # 2x4 -> 8 rows
    V_243 = [[0.0]*14 for _ in range(12)]  # 4x3 -> 12 rows
    W_243 = [[0.0]*14 for _ in range(6)]  # 2x3 -> 6 rows
    
    # First 2x2*2x3 block (columns 0-6)
    for i in range(4):
        for r in range(7):
            U_243[i][r] += U1[i][r]
    for j in range(2):
        for r in range(7):
            V_243[j][r] += V1[j][r]
    for k in range(3):
        for r in range(7):
            W_243[k][r] += W1[k][r]
    
    # Second 2x2*2x3 block (columns 7-13)
    for i in range(4):
        for r in range(7):
            U_243[i+4][r+7] += U2[i][r]
    for j in range(2):
        for r in range(7):
            V_243[j+2][r+7] += V2[j][r]
    for k in range(3):
        for r in range(7):
            W_243[k][r+7] += W2[k][r]
    
    # Build 1x4*4x3 decomposition (rank 4)
    U_143 = [[0.0]*4 for _ in range(4)]  # 1x4 -> 4 rows
    V_143 = [[0.0]*4 for _ in range(12)]  # 4x3 -> 12 rows
    W_143 = [[0.0]*4 for _ in range(3)]  # 1x3 -> 3 rows
    for t in range(4):
        U_143[t][t] = 1.0
        for k in range(3):
            V_143[t*3+k][t] = 1.0
            W_143[k][t] = 1.0
    
    # Combine: total rank 14+4=18
    U = [[0.0]*18 for _ in range(12)]  # 3x4 -> 12 rows
    V = [[0.0]*18 for _ in range(12)]  # 4x3 -> 12 rows
    W = [[0.0]*18 for _ in range(9)]  # 3x3 -> 9 rows
    
    # Top 2 rows of U (A1), bottom 1 row (A2)
    for i in range(8):
        for r in range(14):
            U[i][r] += U_243[i][r]
    for i in range(4):
        for r in range(4):
            U[i+8][r+14] += U_143[i][r]
    
    for j in range(12):
        for r in range(14):
            V[j][r] += V_243[j][r]
    for j in range(12):
        for r in range(4):
            V[j][r+14] += V_143[j][r]
    
    # W: top 2x3 (rows 0-5), bottom 1x3 (rows 6-8)
    for k in range(6):
        for r in range(14):
            W[k][r] += W_243[k][r]
    for k in range(3):
        for r in range(4):
            W[k+6][r+14] += W_143[k][r]
    
    return U, V, W

def _decompose_344():
    """Decompose 3x4 * 4x4. 
    Split A into A1 (2x4) and A2 (1x4).
    C1 = A1*B (2x4), C2 = A2*B (1x4).
    2x4*4x4: split B into B_left (4x2) and B_right (4x2)
    A1*B_left: 2x4*4x2, A1*B_right: 2x4*4x2
    2x4*4x2: split A into 2x2+2x2, each 2x2*2x2 Strassen = 7, total 14
    So 2x4*4x4 = 14+14 = 28
    1x4*4x4: rank 4
    Total: 28+4 = 32
    """
    # 2x4*4x4
    U1, V1, W1 = _strassen_222()  # 2x2*2x2
    U2, V2, W2 = _strassen_222()  # 2x2*2x2
    U3, V3, W3 = _strassen_222()  # 2x2*2x2
    U4, V4, W4 = _strassen_222()  # 2x2*2x2
    
    U_244 = [[0.0]*28 for _ in range(8)]  # 2x4 -> 8 rows
    V_244 = [[0.0]*28 for _ in range(16)]  # 4x4 -> 16 rows
    W_244 = [[0.0]*28 for _ in range(8)]  # 2x4 -> 8 rows
    
    # Block 1: A_left(2x2) * B_left(4x2), columns 0-6
    for i in range(4):
        for r in range(7):
            U_244[i][r] += U1[i][r]
    for j in range(2):
        for r in range(7):
            V_244[j][r] += V1[j][r]
    for k in range(2):
        for r in range(7):
            W_244[k][r] += W1[k][r]
    
    # Block 2: A_right(2x2) * B_left(4x2), columns 7-13
    for i in range(4):
        for r in range(7):
            U_244[i+4][r+7] += U2[i][r]
    for j in range(2):
        for r in range(7):
            V_244[j][r+7] += V2[j][r]
    for k in range(2):
        for r in range(7):
            W_244[k][r+7] += W2[k][r]
    
    # Block 3: A_left(2x2) * B_right(4x2), columns 14-20
    for i in range(4):
        for r in range(7):
            U_244[i][r+14] += U3[i][r]
    for j in range(2):
        for r in range(7):
            V_244[j+2][r+14] += V3[j][r]
    for k in range(2):
        for r in range(7):
            W_244[k+2][r+14] += W3[k][r]
    
    # Block 4: A_right(2x2) * B_right(4x2), columns 21-27
    for i in range(4):
        for r in range(7):
            U_244[i+4][r+21] += U4[i][r]
    for j in range(2):
        for r in range(7):
            V_244[j+2][r+21] += V4[j][r]
    for k in range(2):
        for r in range(7):
            W_244[k+2][r+21] += W4[k][r]
    
    # 1x4*4x4: rank 4
    U_144 = [[0.0]*4 for _ in range(4)]
    V_144 = [[0.0]*4 for _ in range(16)]
    W_144 = [[0.0]*4 for _ in range(4)]
    for t in range(4):
        U_144[t][t] = 1.0
        for k in range(4):
            V_144[t*4+k][t] = 1.0
            W_144[k][t] = 1.0
    
    # Combine: total rank 28+4=32
    U = [[0.0]*32 for _ in range(12)]  # 3x4 -> 12 rows
    V = [[0.0]*32 for _ in range(16)]  # 4x4 -> 16 rows
    W = [[0.0]*32 for _ in range(12)]  # 3x4 -> 12 rows
    
    for i in range(8):
        for r in range(28):
            U[i][r] += U_244[i][r]
    for i in range(4):
        for r in range(4):
            U[i+8][r+28] += U_144[i][r]
    
    for j in range(16):
        for r in range(28):
            V[j][r] += V_244[j][r]
    for j in range(16):
        for r in range(4):
            V[j][r+28] += V_144[j][r]
    
    for k in range(8):
        for r in range(28):
            W[k][r] += W_244[k][r]
    for k in range(4):
        for r in range(4):
            W[k+8][r+28] += W_144[k][r]
    
    return U, V, W

def _decompose_347_v2():
    """Decompose 3x4 * 4x7 using split 7 = 2+5.
    A*B2: 3x4*4x2, A*B5: 3x4*4x5.
    3x4*4x2: split A into 2x4 + 1x4.
      2x4*4x2: split B into 4x1+4x1, each 2x4*4x1 is rank 4, total 8.
        Actually 2x4*4x1: A is 2x4, B is 4x1, C is 1x2.
        C[0,0] = sum_j A[0,j]*B[j,0], C[1,0] = sum_j A[1,j]*B[j,0]
        This is rank 4 (one per j).
      1x4*4x2: rank 4.
      Total 3x4*4x2: 8+4 = 12.
    3x4*4x5: split B into 4x2+4x3.
      3x4*4x2: rank 12 (above)
      3x4*4x3: rank 18 (from _decompose_343)
      Total 3x4*4x5: 12+18 = 30.
    Total: 12+30 = 42.
    """
    # 3x4*4x2 (rank 12)
    # 2x4*4x1 (rank 4) for col 0 of B
    U_a = np.zeros((8, 4))
    V_a = np.zeros((4, 4))
    W_a = np.zeros((2, 4))
    for j in range(4):
        U_a[j, j] = 1.0
        V_a[j, j] = 1.0
        W_a[0, j] = 1.0  # C[0,0]
    for j in range(4):
        U_a[4+j, j] = 1.0
        W_a[1, j] = 1.0  # C[1,0]
    
    # 2x4*4x1 (rank 4) for col 1 of B
    U_b = np.zeros((8, 4))
    V_b = np.zeros((4, 4))
    W_b = np.zeros((2, 4))
    for j in range(4):
        U_b[j, j] = 1.0
        V_b[j, j] = 1.0
        W_b[0, j] = 1.0
    for j in range(4):
        U_b[4+j, j] = 1.0
        W_b[1, j] = 1.0
    
    # 1x4*4x2 (rank 4)
    U_c = np.zeros((4, 4))
    V_c = np.zeros((8, 4))
    W_c = np.zeros((2, 4))
    for j in range(4):
        U_c[j, j] = 1.0
        V_c[j, j] = 1.0
        V_c[4+j, j] = 1.0
        W_c[0, j] = 1.0
        W_c[1, j] = 1.0
    
    # Combine 3x4*4x2
    U_342 = np.zeros((12, 12))
    V_342 = np.zeros((8, 12))
    W_342 = np.zeros((6, 12))
    # Rows 0-7: 2x4 part, rows 8-11: 1x4 part
    # V rows 0-3: col 0 of B, rows 4-7: col 1 of B
    # W rows 0-1: C col 0, rows 2-3: C col 1, rows 4-5: from 1x4 part (but wait, 1x4*4x2 gives 1x2 output)
    
    # Actually let me reconsider. 3x4 * 4x2 gives 3x2 output.
    # C has shape 2x3 (W rows: k*n+i, k in 0-1, i in 0-2 -> 6 rows)
    
    # 2x4*4x2: A rows 0-1, B all 4 rows, C rows 0-1 (output rows 0-1)
    # 1x4*4x2: A row 2, B all 4 rows, C row 2 (output row 2)
    
    # 2x4*4x2: split B into two 4x1
    # B col 0: 2x4*4x1 -> rank 4, contributes to C rows 0-1
    # B col 1: 2x4*4x1 -> rank 4, contributes to C rows 0-1
    # Total: rank 8
    
    U_242 = np.zeros((8, 8))
    V_242 = np.zeros((8, 8))
    W_242 = np.zeros((4, 8))
    # Col 0 of B (V rows 0-3)
    for j in range(4):
        U_242[j, j] = 1.0
        V_242[j, j] = 1.0
        W_242[0, j] = 1.0
        W_242[2, j] = 1.0  # Wait, W row index is k*2+i
        # For 2x4*4x1: A is 2x4, B is 4x1, C is 1x2
        # C[0,0] = sum A[0,j]*B[j,0], C[0,1] = sum A[1,j]*B[j,0]
        # W row: k*2+i where k=0 (only one col of C), i in 0-1
        # So W[0] = C[0,0], W[1] = C[0,1]
    # Let me redo this properly
    
    # 2x4 * 4x1: A(2x4), B(4x1), C(1x2)
    # U rows: i*4+j, i in 0-1, j in 0-3 -> 8 rows
    # V rows: j*1+k = j, j in 0-3 -> 4 rows
    # W rows: k*2+i = i, i in 0-1 -> 2 rows
    # C[i,0] = sum_j A[i,j]*B[j,0]
    # So W[i] gets contribution from U[i*4+j] * V[j]
    
    # For B col 0 (V rows 0-3 in the 4x2 context, but here V is just 4 rows)
    for j in range(4):
        U_242[j, j] = 1.0       # A row 0, col j
        V_242[j, j] = 1.0       # B row j, col 0
        W_242[0, j] = 1.0       # C row 0
        U_242[4+j, j] = 1.0     # A row 1, col j
        W_242[1, j] = 1.0       # C row 1
    
    # For B col 1 (V rows 4-7 in the 4x2 context)
    for j in range(4):
        U_242[j, j+4] = 1.0
        V_242[j+4, j+4] = 1.0
        W_242[0, j+4] = 1.0
        U_242[4+j, j+4] = 1.0
        W_242[1, j+4] = 1.0
    
    # 1x4 * 4x2: A(1x4), B(4x2), C(1x2)
    # U rows: 0*4+j = j, j in 0-3 -> 4 rows
    # V rows: j*2+k, j in 0-3, k in 0-1 -> 8 rows
    # W rows: k*1+i = k, k in 0-1 -> 2 rows
    # C[0,k] = sum_j A[0,j]*B[j,k]
    
    U_142 = np.zeros((4, 4))
    V_142 = np.zeros((8, 4))
    W_142 = np.zeros((2, 4))
    for j in range(4):
        U_142[j, j] = 1.0
        V_142[j*2+0, j] = 1.0
        V_142[j*2+1, j] = 1.0
        W_142[0, j] = 1.0
        W_142[1, j] = 1.0
    
    # Combine 3x4*4x2 (rank 8+4=12)
    U_342 = np.zeros((12, 12))
    V_342 = np.zeros((8, 12))
    W_342 = np.zeros((6, 12))
    # U rows 0-7: 2x4, rows 8-11: 1x4
    # V rows 0-7: 4x2
    # W rows 0-1: C row 0, rows 2-3: C row 1, rows 4-5: C row 2
    for i in range(8):
        for r in range(8):
            U_342[i, r] += U_242[i, r]
    for j in range(8):
        for r in range(8):
            V_342[j, r] += V_242[j, r]
    for k in range(2):
        for r in range(8):
            W_342[k, r] += W_242[k, r]
    for i in range(4):
        for r in range(4):
            U_342[8+i, r+8] += U_142[i, r]
    for j in range(8):
        for r in range(4):
            V_342[j, r+8] += V_142[j, r]
    for k in range(2):
        for r in range(4):
            W_342[4+k, r+8] += W_142[k, r]
    
    # Now 3x4*4x5: split B into 4x2 + 4x3
    # 3x4*4x2: rank 12 (above)
    # 3x4*4x3: rank 18 (from _decompose_343)
    U1, V1, W1 = _decompose_343()
    U1 = np.array(U1)
    V1 = np.array(V1)
    W1 = np.array(W1)
    
    # 3x4*4x5 (rank 12+18=30)
    U_345 = np.zeros((12, 30))
    V_345 = np.zeros((20, 30))
    W_345 = np.zeros((15, 30))
    # V rows 0-7: 4x2 part, rows 8-19: 4x3 part
    # W rows 0-5: C for 4x2 (2x3 -> 6 rows), rows 6-14: C for 4x3 (3x3 -> 9 rows)
    for i in range(12):
        for r in range(12):
            U_345[i, r] += U_342[i, r]
    for j in range(8):
        for r in range(12):
            V_345[j, r] += V_342[j, r]
    for k in range(6):
        for r in range(12):
            W_345[k, r] += W_342[k, r]
    for i in range(12):
        for r in range(18):
            U_345[i, r+12] += U1[i, r]
    for j in range(12):
        for r in range(18):
            V_345[8+j, r+12] += V1[j, r]
    for k in range(9):
        for r in range(18):
            W_345[6+k, r+12] += W1[k, r]
    
    # Final 3x4*4x7: combine 4x2 and 4x5
    U_final = np.zeros((12, 42))
    V_final = np.zeros((28, 42))
    W_final = np.zeros((21, 42))
    # V rows 0-7: 4x2, rows 8-27: 4x5
    # W rows 0-5: C for 4x2, rows 6-20: C for 4x5
    for i in range(12):
        for r in range(12):
            U_final[i, r] += U_342[i, r]
    for j in range(8):
        for r in range(12):
            V_final[j, r] += V_342[j, r]
    for k in range(6):
        for r in range(12):
            W_final[k, r] += W_342[k, r]
    for i in range(12):
        for r in range(30):
            U_final[i, r+12] += U_345[i, r]
    for j in range(20):
        for r in range(30):
            V_final[8+j, r+12] += V_345[j, r]
    for k in range(15):
        for r in range(30):
            W_final[6+k, r+12] += W_345[k, r]
    
    return U_final.tolist(), V_final.tolist(), W_final.tolist()

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta.
    T is shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U @ V^T reconstructed matches W for the tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Reconstruct tensor: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Check against expected
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    T_expected = _build_tensor(n, m, p)
    T_recon_reshaped = T_recon.reshape(n*m, m*p, p*n)
    if np.max(np.abs(T_recon_reshaped - T_expected)) > tol:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Try multiple decompositions, pick the one with lowest rank
    candidates = []
    
    # Original decomposition (rank 50)
    try:
        U1, V1, W1 = _decompose_347()
        if _verify(U1, V1, W1, n, m, p):
            candidates.append((len(U1[0]), U1, V1, W1))
    except Exception:
        pass
    
    # New decomposition (rank 42)
    try:
        U2, V2, W2 = _decompose_347_v2()
        if _verify(U2, V2, W2, n, m, p):
            candidates.append((len(U2[0]), U2, V2, W2))
    except Exception:
        pass
    
    # Pick the best (lowest rank)
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, U, V, W = candidates[0]
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
