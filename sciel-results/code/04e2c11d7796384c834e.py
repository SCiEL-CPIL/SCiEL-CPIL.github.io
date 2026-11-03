import numpy as np
import time

DIMENSIONS = [3, 4, 7]

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

def _decompose_347():
    """Decompose 3x4 * 4x7.
    Split B into B1 (4x3) and B2 (4x4).
    C = A*B1 | A*B2
    A*B1: 3x4*4x3, rank 18
    A*B2: 3x4*4x4, rank 32
    Total: 18+32 = 50
    """
    U1, V1, W1 = _decompose_343()  # rank 18
    U2, V2, W2 = _decompose_344()  # rank 32
    
    total_rank = 18 + 32
    U = [[0.0]*total_rank for _ in range(12)]  # 3x4
    V = [[0.0]*total_rank for _ in range(28)]  # 4x7
    W = [[0.0]*total_rank for _ in range(21)]  # 7x3
    
    # First block (columns 0-17)
    for i in range(12):
        for r in range(18):
            U[i][r] += U1[i][r]
    for j in range(12):
        for r in range(18):
            V[j][r] += V1[j][r]
    for k in range(9):
        for r in range(18):
            W[k][r] += W1[k][r]
    
    # Second block (columns 18-49)
    for i in range(12):
        for r in range(32):
            U[i][r+18] += U2[i][r]
    for j in range(16):
        for r in range(32):
            V[j+12][r+18] += V2[j][r]
    for k in range(12):
        for r in range(32):
            W[k+9][r+18] += W2[k][r]
    
    return U, V, W

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
    
    # Try the block decomposition
    U, V, W = _decompose_347()
    
    # Verify
    if _verify(U, V, W, n, m, p):
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
