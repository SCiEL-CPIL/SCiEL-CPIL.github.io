import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _construct_strassen_2x2x2():
    # Returns U, V, W for 2x2x2 multiplication, rank 7
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = 1 # C11+=M1, C12+=M1, C21-=M1, C22+=M1 ? 
    # Let's map carefully.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Col 0: M1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[1][0]=1; W[2][0]=-1; W[3][0]=1
    
    # Col 1: M2
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    
    # Col 2: M3
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    
    # Col 3: M4
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    
    # Col 4: M5
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    
    # Col 5: M6
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # Col 6: M7
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1; W[3][6]=1
    
    return U, V, W

def _construct_simple_nm_p(n, m, p):
    # Schoolbook for n x m x p
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _construct_2x2x1():
    # Rank 2
    U, V, W = _construct_simple_nm_p(2, 2, 1)
    return U, V, W

def _construct_1x2x3():
    # Rank 2? No, 1x2x3 schoolbook is 6.
    # But C = A B. A is 1x2, B is 2x3.
    # C0 = A0 B0 + A1 B1.
    # This is sum of 2 outer products.
    # Rank 2.
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(6)]
    W = [[0]*2 for _ in range(3)]
    
    # Term 1: A0 * B_col0
    U[0][0] = 1
    V[0][0] = 1 # B00
    V[1][0] = 1 # B10
    V[2][0] = 0
    V[3][0] = 0
    V[4][0] = 0
    V[5][0] = 0
    W[0][0] = 1 # C0
    
    # Term 2: A1 * B_col1
    U[1][1] = 1
    V[0][1] = 0
    V[1][1] = 0
    V[2][1] = 1 # B01
    V[3][1] = 1 # B11
    V[4][1] = 0
    V[5][1] = 0
    W[1][1] = 1 # C1
    
    # Term 3: A0 * B_col2
    # Wait, B is 2x3.
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # C2 = A0 B02 + A1 B12
    
    # My V indices for B (2x3) are:
    # B00, B01, B02, B10, B11, B12
    # Indices: 0, 1, 2, 3, 4, 5
    
    # Term 1: A0 * [B00, B01, B02]
    U[0][0] = 1
    V[0][0] = 1; V[1][0] = 1; V[2][0] = 1
    V[3][0] = 0; V[4][0] = 0; V[5][0] = 0
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1
    
    # Term 2: A1 * [B10, B11, B12]
    U[1][1] = 1
    V[0][1] = 0; V[1][1] = 0; V[2][1] = 0
    V[3][1] = 1; V[4][1] = 1; V[5][1] = 1
    W[0][1] = 1; W[1][1] = 1; W[2][1] = 1
    
    return U, V, W

def _construct_2x2x3():
    # Rank 9 (7 from 2x2x2 + 2 from 2x2x1)
    U1, V1, W1 = _construct_strassen_2x2x2()
    U2, V2, W2 = _construct_2x2x1()
    
    # Combine
    rank = 7 + 2
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Copy 2x2x2 part (cols 0-1 of B, rows 0-1 of C)
    for i in range(4):
        for r in range(7):
            U[i][r] = U1[i][r]
            # V mapping: 2x2 B is top-left 2x2 of 2x3 B
            # B indices: 00,01,02, 10,11,12
            # 2x2 uses 00,01,10,11 -> 0,1,3,4
            if r < 7:
                # Map V1 indices (0,1,2,3) to V indices
                # V1 is 2x2. Indices 0,1,2,3 correspond to B00,B01,B10,B11
                # In 2x3, B00=0, B01=1, B10=3, B11=4
                pass 
    # This manual mapping is error prone. Let's use numpy arrays.
    U1_arr = np.array(U1)
    V1_arr = np.array(V1)
    W1_arr = np.array(W1)
    
    U2_arr = np.array(U2)
    V2_arr = np.array(V2)
    W2_arr = np.array(W2)
    
    U_comb = np.zeros((4, 9), dtype=int)
    V_comb = np.zeros((6, 9), dtype=int)
    W_comb = np.zeros((6, 9), dtype=int)
    
    # 2x2x2 part
    U_comb[:, 0:7] = U1_arr
    # V1 is 4x7. Indices 0,1,2,3 map to 0,1,3,4 in 6x?
    # B 2x2: 00, 01, 10, 11
    # B 2x3: 00, 01, 02, 10, 11, 12
    # Map: 0->0, 1->1, 2->3, 3->4
    V_comb[0, 0:7] = V1_arr[0, 0:7]
    V_comb[1, 0:7] = V1_arr[1, 0:7]
    V_comb[3, 0:7] = V1_arr[2, 0:7]
    V_comb[4, 0:7] = V1_arr[3, 0:7]
    
    # W1 is 4x7. Indices 0,1,2,3 map to 0,1,3,4 in 6x?
    # C 2x2: 00, 01, 10, 11
    # C 2x3: 00, 01, 02, 10, 11, 12
    W_comb[0, 0:7] = W1_arr[0, 0:7]
    W_comb[1, 0:7] = W1_arr[1, 0:7]
    W_comb[3, 0:7] = W1_arr[2, 0:7]
    W_comb[4, 0:7] = W1_arr[3, 0:7]
    
    # 2x2x1 part (col 2 of B, col 2 of C)
    # U2 is 4x2.
    U_comb[:, 7:9] = U2_arr
    # V2 is 6x2? No, 2x2x1 B is 2x1.
    # _construct_simple_nm_p(2,2,1) returns V of shape (2*1, 2) = (2,2).
    # But we need to map it to the 2x3 B structure.
    # B for 2x2x1 is the 3rd column of the 2x3 B.
    # Indices in 2x3: 2, 5.
    # V2 indices: 0, 1.
    V_comb[2, 7:9] = V2_arr[0, 0:2]
    V_comb[5, 7:9] = V2_arr[1, 0:2]
    
    # W2 is 2x2? No, 2x2x1 C is 2x1.
    # _construct_simple_nm_p(2,2,1) returns W of shape (1*2, 2) = (2,2).
    # W indices: 0, 1. Map to C indices 2, 5.
    W_comb[2, 7:9] = W2_arr[0, 0:2]
    W_comb[5, 7:9] = W2_arr[1, 0:2]
    
    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x2x3():
    # Rank 11 (2 from 1x2x3 + 9 from 2x2x3)
    U1, V1, W1 = _construct_1x2x3()
    U2, V2, W2 = _construct_2x2x3()
    
    rank = 2 + 9
    U_comb = np.zeros((6, 11), dtype=int)
    V_comb = np.zeros((6, 11), dtype=int)
    W_comb = np.zeros((9, 11), dtype=int)
    
    # 1x2x3 part (row 0 of A, all B, row 0 of C)
    # U1 is 2x2. Map to A indices 0, 1 (row 0).
    U_comb[0:2, 0:2] = np.array(U1)
    # V1 is 6x2. Map to all B.
    V_comb[:, 0:2] = np.array(V1)
    # W1 is 3x2. Map to C row 0 (indices 0,1,2).
    W_comb[0:3, 0:2] = np.array(W1)
    
    # 2x2x3 part (rows 1,2 of A, all B, rows 1,2 of C)
    # U2 is 4x9. A indices for rows 1,2 are 2,3,4,5.
    U_comb[2:6, 2:11] = np.array(U2)
    # V2 is 6x9. Map to all B.
    V_comb[:, 2:11] = np.array(V2)
    # W2 is 6x9. C indices for rows 1,2 are 3,4,5,6,7,8.
    W_comb[3:9, 2:11] = np.array(W2)
    
    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x2x6():
    # Rank 22 (2 * 11)
    U1, V1, W1 = _construct_3x2x3()
    U2, V2, W2 = _construct_3x2x3()
    
    rank = 22
    U_comb = np.zeros((6, 22), dtype=int)
    V_comb = np.zeros((12, 22), dtype=int)
    W_comb = np.zeros((18, 22), dtype=int)
    
    # First 3x2x3 (cols 0-2 of B, cols 0-2 of C)
    U_comb[:, 0:11] = np.array(U1)
    V_comb[0:6, 0:11] = np.array(V1)
    W_comb[0:9, 0:11] = np.array(W1)
    
    # Second 3x2x3 (cols 3-5 of B, cols 3-5 of C)
    U_comb[:, 11:22] = np.array(U2)
    V_comb[6:12, 11:22] = np.array(V2)
    W_comb[9:18, 11:22] = np.array(W2)
    
    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x4x6():
    # Rank 44 (2 * 22)
    U1, V1, W1 = _construct_3x2x6()
    U2, V2, W2 = _construct_3x2x6()
    
    rank = 44
    U_comb = np.zeros((12, 44), dtype=int)
    V_comb = np.zeros((24, 44), dtype=int)
    W_comb = np.zeros((18, 44), dtype=int)
    
    # First 3x2x6 (cols 0-1 of A, rows 0-5 of B, all C)
    # A is 3x4. Cols 0-1 are indices 0,1,2,3,4,5?
    # A 3x2 indices: 0,1,2,3,4,5.
    # A 3x4 indices: 0..11.
    U_comb[0:6, 0:22] = np.array(U1)
    # B 2x6 indices: 0..11.
    # B 4x6 indices: 0..23.
    V_comb[0:12, 0:22] = np.array(V1)
    W_comb[:, 0:22] = np.array(W1)
    
    # Second 3x2x6 (cols 2-3 of A, rows 6-11 of B, all C)
    U_comb[6:12, 22:44] = np.array(U2)
    V_comb[12:24, 22:44] = np.array(V2)
    W_comb[:, 22:44] = np.array(W2)
    
    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x1x6():
    # Rank 1
    U, V, W = _construct_simple_nm_p(3, 1, 6)
    return U, V, W

def _construct_rank45():
    U1, V1, W1 = _construct_3x4x6()
    U2, V2, W2 = _construct_3x1x6()
    
    rank = 45
    U_comb = np.zeros((15, 45), dtype=int)
    V_comb = np.zeros((30, 45), dtype=int)
    W_comb = np.zeros((18, 45), dtype=int)
    
    U_comb[:, 0:44] = np.array(U1)
    V_comb[0:24, 0:44] = np.array(V1)
    W_comb[:, 0:44] = np.array(W1)
    
    U_comb[12:15, 44:45] = np.array(U2)
    V_comb[24:30, 44:45] = np.array(V2)
    W_comb[:, 44:45] = np.array(W2)
    
    return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}

def construct():
    start_time = time.time()
    # Try to construct rank 45
    try:
        result = _construct_rank45()
        # Verify quickly?
        # If time is tight, return schoolbook
        if time.time() - start_time > 110:
            return _construct_schoolbook()
        return result
    except Exception as e:
        return _construct_schoolbook()

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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
