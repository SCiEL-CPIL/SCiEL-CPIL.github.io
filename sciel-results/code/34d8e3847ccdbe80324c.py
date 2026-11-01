"""Matrix multiplication 3x3x3 rank-27 decomposition (fallback) or rank-26 attempt."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts Rank 26, falls back to Rank 27.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 3
    
    # Rank 27 Direct
    rank27 = 27
    U27 = [[0] * rank27 for _ in range(9)]
    V27 = [[0] * rank27 for _ in range(9)]
    W27 = [[0] * rank27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U27[i * 3 + j][col] = 1
                V27[j * 3 + k][col] = 1
                W27[k * 3 + i][col] = 1
                col += 1
    
    # Try Rank 26
    rank26 = 26
    U = [[0] * rank26 for _ in range(9)]
    V = [[0] * rank26 for _ in range(9)]
    W = [[0] * rank26 for _ in range(9)]
    
    # Strassen terms 0-6
    # Using a reliable Strassen definition
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[4][0] += 1
    
    # M2: (A10+A11)B00
    U[3][1] += 1; U[4][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[4][1] -= 1
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[4][2] -= 1
    W[3][2] += 1; W[4][2] += 1
    
    # M4: A11(B10-B00)
    U[4][3] += 1
    V[3][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    # M5: (A00+A10)B11
    U[0][4] += 1; U[3][4] += 1
    V[4][4] += 1
    W[0][4] -= 1; W[3][4] += 1
    
    # M6: (A01-A00)(B00+B10)
    U[1][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[3][5] += 1
    W[4][5] += 1
    
    # M7: (A11-A10)(B01+B11)
    U[4][6] += 1; U[3][6] -= 1
    V[1][6] += 1; V[4][6] += 1
    W[0][6] -= 1
    
    # Cross terms 7-25
    # 7: A02 B20 -> C00
    U[2][7] += 1; V[6][7] += 1; W[0][7] += 1
    # 8: A02 B21 -> C01
    U[2][8] += 1; V[7][8] += 1; W[3][8] += 1
    # 9: A12 B20 -> C10
    U[5][9] += 1; V[6][9] += 1; W[1][9] += 1
    # 10: A12 B21 -> C11
    U[5][10] += 1; V[7][10] += 1; W[4][10] += 1
    # 11: A00 B02 -> C02
    U[0][11] += 1; V[2][11] += 1; W[6][11] += 1
    # 12: A01 B12 -> C02
    U[1][12] += 1; V[5][12] += 1; W[6][12] += 1
    # 13: A10 B02 -> C12
    U[3][13] += 1; V[2][13] += 1; W[7][13] += 1
    # 14: A11 B12 -> C12
    U[4][14] += 1; V[5][14] += 1; W[7][14] += 1
    # 15: A20 B00 -> C20
    U[6][15] += 1; V[0][15] += 1; W[2][15] += 1
    # 16: A20 B01 -> C21
    U[6][16] += 1; V[1][16] += 1; W[5][16] += 1
    # 17: A21 B10 -> C20
    U[7][17] += 1; V[3][17] += 1; W[2][17] += 1
    # 18: A21 B11 -> C21
    U[7][18] += 1; V[4][18] += 1; W[5][18] += 1
    # 19: A02 B22 -> C02
    U[2][19] += 1; V[8][19] += 1; W[6][19] += 1
    # 20: A12 B22 -> C12
    U[5][20] += 1; V[8][20] += 1; W[7][20] += 1
    # 21: A20 B02 -> C22
    U[6][21] += 1; V[2][21] += 1; W[8][21] += 1
    # 22: A21 B12 -> C22
    U[7][22] += 1; V[5][22] += 1; W[8][22] += 1
    # 23: A22 B20 -> C20
    U[8][23] += 1; V[6][23] += 1; W[2][23] += 1
    # 24: A22 B21 -> C21
    U[8][24] += 1; V[7][24] += 1; W[5][24] += 1
    # 25: A22 B22 -> C22
    U[8][25] += 1; V[8][25] += 1; W[8][25] += 1
    
    # Verify Rank 26
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(rank26):
                    s += U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U27, "V": V27, "W": W27}
# EVOLVE-BLOCK-END
