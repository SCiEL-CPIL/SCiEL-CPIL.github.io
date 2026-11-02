import time
import numpy as np

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not (isinstance(val, (int, float)) and (val * 2 == int(val * 2))):
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1..M7
    # U (A)
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, -1], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    # V (B)
    V_S = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 0, 1],  # B01
        [1, 0, 1, 0, 0, 1, 0],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    # W (C)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 + M7
    W_S = [
        [1, 0, 0, -1, 1, 0, -1], # C00
        [0, 0, 1, 1, 0, 0, 0],   # C01
        [0, 1, 0, 0, 0, 0, 1],   # C10
        [1, 0, -1, 0, 0, 1, 1]   # C11
    ]
    
    # Schoolbook 2x1x2 coefficients
    # P1 = A0 * B0
    # P2 = A0 * B1
    # P3 = A1 * B0
    # P4 = A1 * B1
    U_SB = [
        [1, 1, 0, 0], # A0
        [0, 0, 1, 1]  # A1
    ]
    V_SB = [
        [1, 0, 1, 0], # B0
        [0, 1, 0, 1]  # B1
    ]
    W_SB = [
        [1, 0, 0, 0], # C0
        [0, 1, 0, 0], # C1
        [0, 0, 1, 0], # C2
        [0, 0, 0, 1]  # C3
    ]
    
    # Part 1: A_left (2x2) * B_top (2x6)
    # B_top split into 3 blocks of 2x2
    for b in range(3):
        col_start = b * 7
        # A_left indices: i in 0,1; j in 0,1
        # U rows: 0, 1, 5, 6
        U[0][col_start:col_start+7] = U_S[0]
        U[1][col_start:col_start+7] = U_S[1]
        U[5][col_start:col_start+7] = U_S[2]
        U[6][col_start:col_start+7] = U_S[3]
        
        # B_top indices: j in 0,1; k in 2b, 2b+1
        # V rows: 2*6+2b, 2*6+2b+1, 1*6+2b, 1*6+2b+1
        # V rows: 12+2b? No. j=0 -> 0..5. j=1 -> 6..11.
        # V[0*6+2b] = V[2b]
        # V[0*6+2b+1] = V[2b+1]
        # V[1*6+2b] = V[6+2b]
        # V[1*6+2b+1] = V[7+2b]
        V[2*b][col_start:col_start+7] = V_S[0]
        V[2*b+1][col_start:col_start+7] = V_S[1]
        V[6+2*b][col_start:col_start+7] = V_S[2]
        V[7+2*b][col_start:col_start+7] = V_S[3]
        
        # W indices: k in 2b, 2b+1; i in 0,1
        # W rows: 2*(2b)+0, 2*(2b)+1, 2*(2b+1)+0, 2*(2b+1)+1
        # W[4b], W[4b+1], W[4b+2], W[4b+3]
        W[4*b][col_start:col_start+7] = W_S[0]
        W[4*b+2][col_start:col_start+7] = W_S[1]
        W[4*b+1][col_start:col_start+7] = W_S[2]
        W[4*b+3][col_start:col_start+7] = W_S[3]
        
    # Part 2: A_right (2x3) * B_bottom (3x6)
    # B_bottom split into 3 blocks of 3x2
    # Each block uses Rank 11 (7 Strassen + 4 Schoolbook)
    for b in range(3):
        col_start = 21 + b * 11
        
        # Strassen part (cols 0-6 of this block)
        # A_right uses j in 2,3 for Strassen part
        # U rows: 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
        U[2][col_start:col_start+7] = U_S[0]
        U[3][col_start:col_start+7] = U_S[1]
        U[7][col_start:col_start+7] = U_S[2]
        U[8][col_start:col_start+7] = U_S[3]
        
        # B_bottom uses j in 2,3 for Strassen part
        # V rows: 2*6+2b, 2*6+2b+1, 3*6+2b, 3*6+2b+1
        V[12+2*b][col_start:col_start+7] = V_S[0]
        V[13+2*b][col_start:col_start+7] = V_S[1]
        V[18+2*b][col_start:col_start+7] = V_S[2]
        V[19+2*b][col_start:col_start+7] = V_S[3]
        
        # W is same as before
        W[4*b][col_start:col_start+7] = W_S[0]
        W[4*b+2][col_start:col_start+7] = W_S[1]
        W[4*b+1][col_start:col_start+7] = W_S[2]
        W[4*b+3][col_start:col_start+7] = W_S[3]
        
        # Schoolbook part (cols 7-10 of this block)
        sb_start = col_start + 7
        # A_right uses j in 4 for Schoolbook part
        # U rows: 0*5+4=4, 1*5+4=9
        U[4][sb_start:sb_start+4] = U_SB[0]
        U[9][sb_start:sb_start+4] = U_SB[1]
        
        # B_bottom uses j in 4 for Schoolbook part
        # V rows: 4*6+2b, 4*6+2b+1
        V[24+2*b][sb_start:sb_start+4] = V_SB[0]
        V[25+2*b][sb_start:sb_start+4] = V_SB[1]
        
        # W is same as before
        W[4*b][sb_start:sb_start+4] = W_SB[0]
        W[4*b+2][sb_start:sb_start+4] = W_SB[1]
        W[4*b+1][sb_start:sb_start+4] = W_SB[2]
        W[4*b+3][sb_start:sb_start+4] = W_SB[3]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
