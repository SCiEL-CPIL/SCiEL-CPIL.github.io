"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Rank 7.
    U: 4x7, V: 4x7, W: 4x7
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A11*(B11-B21)
    U[0][2] = 1
    V[0][2] = 1; V[1][2] = -1
    W[1][2] = 1
    
    # M4 = A22*(B21-B22)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _decompose_2x2x7():
    """
    Decompose 2x2 * 2x7.
    Rank 25.
    U: 4x25, V: 14x25, W: 14x25
    U[i*2+j, r]
    V[j*7+k, r]
    W[k*2+i, r]
    """
    rank = 25
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(14)]
    W = [[0]*rank for _ in range(14)]
    
    base = _strassen_2x2()
    
    # Three Strassen blocks for k in {0,1}, {2,3}, {4,5}
    for block_idx in range(3):
        k_offset = block_idx * 2
        col_offset = block_idx * 7
        
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    U[i*2+j][col_offset + r] = base["U"][i*2+j][r]
            
            # V
            for j in range(2):
                for k in range(2):
                    V[j*7 + (k_offset + k)][col_offset + r] = base["V"][j*2+k][r]
            
            # W
            for i in range(2):
                for k in range(2):
                    W[(k_offset + k)*2 + i][col_offset + r] = base["W"][k*2+i][r]
    
    # One Schoolbook block for k in {6}
    # 2x2 * 2x1
    # Rank 4
    col_offset = 21
    for i in range(2):
        for j in range(2):
            r = col_offset + (i*2 + j)
            U[i*2+j][r] = 1
            V[j*7 + 6][r] = 1
            W[6*2 + i][r] = 1
            
    return {"U": U, "V": V, "W": W}

def _decompose_1x2x7():
    """
    Decompose 1x2 * 2x7.
    Rank 14.
    U: 2x14, V: 14x14, W: 7x14
    U[j, r]
    V[j*7+k, r]
    W[k*1+0, r] -> W[k, r]
    """
    rank = 14
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(14)]
    W = [[0]*rank for _ in range(7)]
    
    for j in range(2):
        for k in range(7):
            r = j*7 + k
            U[j][r] = 1
            V[j*7+k][r] = 1
            W[k][r] = 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS # 3, 4, 7
    
    # Total rank 78
    total_rank = 78
    U = [[0]*total_rank for _ in range(n*m)] # 12 x 78
    V = [[0]*total_rank for _ in range(m*p)] # 28 x 78
    W = [[0]*total_rank for _ in range(p*n)] # 21 x 78
    
    # Block 1: A11 (2x2) * B_top (2x7)
    # A rows 0,1; B rows 0,1; C cols 0-6, rows 0,1
    d1 = _decompose_2x2x7()
    for r in range(25):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][r] = d1["U"][i*2+j][r]
        for j in range(2):
            for k in range(7):
                V[j*7 + k][r] = d1["V"][j*7+k][r]
        for i in range(2):
            for k in range(7):
                W[k*3 + i][r] = d1["W"][k*2+i][r]
                
    # Block 2: A12 (2x2) * B_bot (2x7)
    # A rows 0,1; B rows 2,3; C cols 0-6, rows 0,1
    # U rows 4-7
    # V rows 14-27
    # W rows k*3 + i (i in 0,1)
    d2 = _decompose_2x2x7()
    col_off = 25
    for r in range(25):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2)][col_off + r] = d2["U"][i*2+j][r]
        for j in range(2):
            for k in range(7):
                V[(j+2)*7 + k][col_off + r] = d2["V"][j*7+k][r]
        for i in range(2):
            for k in range(7):
                W[k*3 + i][col_off + r] = d2["W"][k*2+i][r]
                
    # Block 3: A2 (1x2) * B_top (2x7)
    # A row 2, j=0,1; B rows 0,1; C cols 0-6, row 2
    d3 = _decompose_1x2x7()
    col_off = 50
    for r in range(14):
        for j in range(2):
            U[2*4 + j][col_off + r] = d3["U"][j][r]
        for j in range(2):
            for k in range(7):
                V[j*7 + k][col_off + r] = d3["V"][j*7+k][r]
        for k in range(7):
            W[k*3 + 2][col_off + r] = d3["W"][k][r]
            
    # Block 4: A2 (1x2) * B_bot (2x7)
    # A row 2, j=2,3; B rows 2,3; C cols 0-6, row 2
    d4 = _decompose_1x2x7()
    col_off = 64
    for r in range(14):
        for j in range(2):
            U[2*4 + (j+2)][col_off + r] = d4["U"][j][r]
        for j in range(2):
            for k in range(7):
                V[(j+2)*7 + k][col_off + r] = d4["V"][j*7+k][r]
        for k in range(7):
            W[k*3 + 2][col_off + r] = d4["W"][k][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
