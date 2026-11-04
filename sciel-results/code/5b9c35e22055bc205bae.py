import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2.
    U: shape (4, 7) - rows indexed by (i,j) for i,j in {0,1}
    V: shape (4, 7) - rows indexed by (j,k) for j,k in {0,1}
    W: shape (4, 7) - rows indexed by (k,i) for k,i in {0,1}
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_2x2_2x1():
    """
    Return the schoolbook rank-4 decomposition for 2x2 * 2x1.
    U: shape (4, 4) - rows indexed by (i,j) for i,j in {0,1}
    V: shape (2, 4) - rows indexed by (j,0) for j in {0,1}
    W: shape (2, 4) - rows indexed by (0,i) for i in {0,1}
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # C00 = A00*B00 + A01*B10
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    
    # C10 = A10*B00 + A11*B10
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def _add_2x2_2x7(U, V, W, a_rows, b_rows, c_rows, col_offset):
    """
    Add a 2x2 * 2x7 decomposition at the given row indices and column offset.
    a_rows: list of 4 global U row indices
    b_rows: list of 14 global V row indices (j*7+k)
    c_rows: list of 14 global W row indices (k*2+i) -- wait, local W index is k*2+i.
    """
    # Split B into three parts: B[:, 0:2], B[:, 2:4], B[:, 4:7]
    # Part 1: 2x2 * 2x2 (Strassen)
    b_part1 = [b_rows[j*7 + k] for j in range(2) for k in range(2)]
    c_part1 = [c_rows[k*2 + i] for k in range(2) for i in range(2)]
    U_s, V_s, W_s = _strassen_2x2_2x2()
    for idx, row in enumerate(a_rows):
        for c in range(7):
            U[row][col_offset + c] += U_s[idx][c]
    for idx, row in enumerate(b_part1):
        for c in range(7):
            V[row][col_offset + c] += V_s[idx][c]
    for idx, row in enumerate(c_part1):
        for c in range(7):
            W[row][col_offset + c] += W_s[idx][c]
    
    col_offset += 7
    
    # Part 2: 2x2 * 2x2 (Strassen)
    b_part2 = [b_rows[j*7 + k] for j in range(2) for k in range(2, 4)]
    c_part2 = [c_rows[k*2 + i] for k in range(2, 4) for i in range(2)]
    U_s, V_s, W_s = _strassen_2x2_2x2()
    for idx, row in enumerate(a_rows):
        for c in range(7):
            U[row][col_offset + c] += U_s[idx][c]
    for idx, row in enumerate(b_part2):
        for c in range(7):
            V[row][col_offset + c] += V_s[idx][c]
    for idx, row in enumerate(c_part2):
        for c in range(7):
            W[row][col_offset + c] += W_s[idx][c]
    
    col_offset += 7
    
    # Part 3: 2x2 * 2x3
    # Split into 2x2 * 2x2 (Strassen) and 2x2 * 2x1 (Schoolbook)
    b_part3a = [b_rows[j*7 + k] for j in range(2) for k in range(4, 6)]
    c_part3a = [c_rows[k*2 + i] for k in range(4, 6) for i in range(2)]
    U_s, V_s, W_s = _strassen_2x2_2x2()
    for idx, row in enumerate(a_rows):
        for c in range(7):
            U[row][col_offset + c] += U_s[idx][c]
    for idx, row in enumerate(b_part3a):
        for c in range(7):
            V[row][col_offset + c] += V_s[idx][c]
    for idx, row in enumerate(c_part3a):
        for c in range(7):
            W[row][col_offset + c] += W_s[idx][c]
    
    col_offset += 7
    
    b_part3b = [b_rows[j*7 + 6] for j in range(2)]
    c_part3b = [c_rows[6*2 + i] for i in range(2)]
    U_s, V_s, W_s = _schoolbook_2x2_2x1()
    for idx, row in enumerate(a_rows):
        for c in range(4):
            U[row][col_offset + c] += U_s[idx][c]
    for idx, row in enumerate(b_part3b):
        for c in range(4):
            V[row][col_offset + c] += V_s[idx][c]
    for idx, row in enumerate(c_part3b):
        for c in range(4):
            W[row][col_offset + c] += W_s[idx][c]
    
    col_offset += 4
    
    return col_offset

def construct():
    n, m, p = 4, 4, 7
    rank = 100
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Block 1: A11 * B1 -> C11
    # A11: i in {0,1}, j in {0,1} -> U rows: i*4+j
    a_rows_1 = [0, 1, 4, 5]
    # B1: j in {0,1}, k in {0..6} -> V rows: j*7+k
    b_rows_1 = [j*7 + k for j in range(2) for k in range(7)]
    # C11: i in {0,1}, k in {0..6} -> W rows: k*4+i
    c_rows_1 = [k*4 + i for k in range(7) for i in range(2)]
    
    col = _add_2x2_2x7(U, V, W, a_rows_1, b_rows_1, c_rows_1, col)
    
    # Block 2: A12 * B2 -> C12
    # A12: i in {0,1}, j in {2,3} -> U rows: i*4+j
    a_rows_2 = [2, 3, 6, 7]
    # B2: j in {2,3}, k in {0..6} -> V rows: j*7+k
    b_rows_2 = [j*7 + k for j in range(2, 4) for k in range(7)]
    # C12: i in {0,1}, k in {0..6} -> W rows: k*4+i
    c_rows_2 = [k*4 + i for k in range(7) for i in range(2)]
    
    col = _add_2x2_2x7(U, V, W, a_rows_2, b_rows_2, c_rows_2, col)
    
    # Block 3: A21 * B1 -> C21
    # A21: i in {2,3}, j in {0,1} -> U rows: i*4+j
    a_rows_3 = [8, 9, 12, 13]
    # B1: j in {0,1}, k in {0..6} -> V rows: j*7+k
    b_rows_3 = [j*7 + k for j in range(2) for k in range(7)]
    # C21: i in {2,3}, k in {0..6} -> W rows: k*4+i
    c_rows_3 = [k*4 + i for k in range(7) for i in range(2, 4)]
    
    col = _add_2x2_2x7(U, V, W, a_rows_3, b_rows_3, c_rows_3, col)
    
    # Block 4: A22 * B2 -> C22
    # A22: i in {2,3}, j in {2,3} -> U rows: i*4+j
    a_rows_4 = [10, 11, 14, 15]
    # B2: j in {2,3}, k in {0..6} -> V rows: j*7+k
    b_rows_4 = [j*7 + k for j in range(2, 4) for k in range(7)]
    # C22: i in {2,3}, k in {0..6} -> W rows: k*4+i
    c_rows_4 = [k*4 + i for k in range(7) for i in range(2, 4)]
    
    col = _add_2x2_2x7(U, V, W, a_rows_4, b_rows_4, c_rows_4, col)
    
    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return _schoolbook() # Fallback
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
