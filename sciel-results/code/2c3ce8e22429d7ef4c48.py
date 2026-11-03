import time
from fractions import Fraction as F

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication with rank 7.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U: 0=A00, 1=A01, 2=A10, 3=A11
    V: 0=B00, 1=B01, 2=B10, 3=B11
    W: 0=C00, 1=C01, 2=C10, 3=C11
    """
    U = [[F(0) for _ in range(7)] for _ in range(4)]
    V = [[F(0) for _ in range(7)] for _ in range(4)]
    W = [[F(0) for _ in range(7)] for _ in range(4)]
    
    # P1 = (A00+A11)(B00+B11)
    U[0][0] = F(1); U[3][0] = F(1)
    V[0][0] = F(1); V[3][0] = F(1)
    W[0][0] = F(1); W[3][0] = F(1)
    
    # P2 = (A10+A11)B00
    U[2][1] = F(1); U[3][1] = F(1)
    V[0][1] = F(1)
    W[2][1] = F(1); W[3][1] = F(-1)
    
    # P3 = A00(B10-B11)
    U[0][2] = F(1)
    V[2][2] = F(1); V[3][2] = F(-1)
    W[1][2] = F(1); W[3][2] = F(1)
    
    # P4 = A11(B01-B00)
    U[3][3] = F(1)
    V[1][3] = F(1); V[0][3] = F(-1)
    W[0][3] = F(1); W[2][3] = F(1)
    
    # P5 = (A00+A01)B11
    U[0][4] = F(1); U[1][4] = F(1)
    V[3][4] = F(1)
    W[0][4] = F(-1); W[1][4] = F(1)
    
    # P6 = (A10-A00)(B00+B01)
    U[2][5] = F(1); U[0][5] = F(-1)
    V[0][5] = F(1); V[1][5] = F(1)
    W[3][5] = F(1)
    
    # P7 = (A01-A11)(B10+B11)
    U[1][6] = F(1); U[3][6] = F(-1)
    V[2][6] = F(1); V[3][6] = F(1)
    W[0][6] = F(1)
    
    return U, V, W

def _schoolbook_1x2x2():
    """
    Returns U, V, W for 1x2 matrix times 2x2 matrix with rank 4.
    U: 2x4, V: 4x4, W: 2x4
    Indices:
    U: 0=A0, 1=A1
    V: 0=B00, 1=B01, 2=B10, 3=B11
    W: 0=C0, 1=C1
    """
    U = [[F(0) for _ in range(4)] for _ in range(2)]
    V = [[F(0) for _ in range(4)] for _ in range(4)]
    W = [[F(0) for _ in range(4)] for _ in range(2)]
    
    # T1 = A0 B00 -> C0
    U[0][0] = F(1)
    V[0][0] = F(1)
    W[0][0] = F(1)
    
    # T2 = A0 B01 -> C1
    U[0][1] = F(1)
    V[1][1] = F(1)
    W[1][1] = F(1)
    
    # T3 = A1 B10 -> C0
    U[1][2] = F(1)
    V[2][2] = F(1)
    W[0][2] = F(1)
    
    # T4 = A1 B11 -> C1
    U[1][3] = F(1)
    V[3][3] = F(1)
    W[1][3] = F(1)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # U size: n*m = 12
    # V size: m*p = 16
    # W size: p*n = 12
    
    rank = 44
    U = [[F(0) for _ in range(rank)] for _ in range(12)]
    V = [[F(0) for _ in range(rank)] for _ in range(16)]
    W = [[F(0) for _ in range(rank)] for _ in range(12)]
    
    S_U, S_V, S_W = _strassen_2x2()
    B_U, B_V, B_W = _schoolbook_1x2x2()
    
    col = 0
    
    # --- Block 1: A_top (2x4) * B (4x4) -> C_top (2x4) ---
    # A_top = [A_TL A_TR]
    # A_TL indices in U: 0,1,2,3
    # A_TR indices in U: 4,5,6,7
    # B = [[B_TL B_TR]
    #      [B_BL B_BR]]
    # B_TL indices in V: 0,1,2,3
    # B_TR indices in V: 4,5,6,7
    # B_BL indices in V: 8,9,10,11
    # B_BR indices in V: 12,13,14,15
    # C_top = [C_TL C_TR]
    # C_TL indices in W: 0,1,2,3
    # C_TR indices in W: 4,5,6,7
    
    # C_TL = A_TL B_TL + A_TR B_BL
    # Term 1: A_TL B_TL (Strassen)
    for r in range(7):
        for i in range(4):
            if S_U[i][r] != 0: U[0+i][col] += S_U[i][r]
            if S_V[i][r] != 0: V[0+i][col] += S_V[i][r]
            if S_W[i][r] != 0: W[0+i][col] += S_W[i][r]
        col += 1
        
    # Term 2: A_TR B_BL (Strassen)
    for r in range(7):
        for i in range(4):
            if S_U[i][r] != 0: U[4+i][col] += S_U[i][r]
            if S_V[i][r] != 0: V[8+i][col] += S_V[i][r]
            if S_W[i][r] != 0: W[0+i][col] += S_W[i][r]
        col += 1
        
    # C_TR = A_TL B_TR + A_TR B_BR
    # Term 3: A_TL B_TR (Strassen)
    for r in range(7):
        for i in range(4):
            if S_U[i][r] != 0: U[0+i][col] += S_U[i][r]
            if S_V[i][r] != 0: V[4+i][col] += S_V[i][r]
            if S_W[i][r] != 0: W[4+i][col] += S_W[i][r]
        col += 1
        
    # Term 4: A_TR B_BR (Strassen)
    for r in range(7):
        for i in range(4):
            if S_U[i][r] != 0: U[4+i][col] += S_U[i][r]
            if S_V[i][r] != 0: V[12+i][col] += S_V[i][r]
            if S_W[i][r] != 0: W[4+i][col] += S_W[i][r]
        col += 1

    # --- Block 2: A_bot (1x4) * B (4x4) -> C_bot (1x4) ---
    # A_bot = [A_BL A_BR]
    # A_BL indices in U: 8,9
    # A_BR indices in U: 10,11
    # C_bot = [C_BL C_BR]
    # C_BL indices in W: 8,9
    # C_BR indices in W: 10,11
    
    # C_BL = A_BL B_TL + A_BR B_BL
    # Term 5: A_BL B_TL (Schoolbook 1x2x2)
    # A_BL is 1x2. B_TL is 2x2.
    # U indices for A_BL: 8,9.
    # V indices for B_TL: 0,1,2,3.
    # W indices for C_BL: 8,9.
    for r in range(4):
        for i in range(2):
            if B_U[i][r] != 0: U[8+i][col] += B_U[i][r]
        for i in range(4):
            if B_V[i][r] != 0: V[0+i][col] += B_V[i][r]
        for i in range(2):
            if B_W[i][r] != 0: W[8+i][col] += B_W[i][r]
        col += 1
        
    # Term 6: A_BR B_BL (Schoolbook 1x2x2)
    # A_BR indices: 10,11.
    # B_BL indices: 8,9,10,11.
    # C_BL indices: 8,9.
    for r in range(4):
        for i in range(2):
            if B_U[i][r] != 0: U[10+i][col] += B_U[i][r]
        for i in range(4):
            if B_V[i][r] != 0: V[8+i][col] += B_V[i][r]
        for i in range(2):
            if B_W[i][r] != 0: W[8+i][col] += B_W[i][r]
        col += 1
        
    # C_BR = A_BL B_TR + A_BR B_BR
    # Term 7: A_BL B_TR (Schoolbook 1x2x2)
    # A_BL indices: 8,9.
    # B_TR indices: 4,5,6,7.
    # C_BR indices: 10,11.
    for r in range(4):
        for i in range(2):
            if B_U[i][r] != 0: U[8+i][col] += B_U[i][r]
        for i in range(4):
            if B_V[i][r] != 0: V[4+i][col] += B_V[i][r]
        for i in range(2):
            if B_W[i][r] != 0: W[10+i][col] += B_W[i][r]
        col += 1
        
    # Term 8: A_BR B_BR (Schoolbook 1x2x2)
    # A_BR indices: 10,11.
    # B_BR indices: 12,13,14,15.
    # C_BR indices: 10,11.
    for r in range(4):
        for i in range(2):
            if B_U[i][r] != 0: U[10+i][col] += B_U[i][r]
        for i in range(4):
            if B_V[i][r] != 0: V[12+i][col] += B_V[i][r]
        for i in range(2):
            if B_W[i][r] != 0: W[10+i][col] += B_W[i][r]
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
