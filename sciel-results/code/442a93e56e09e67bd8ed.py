import time
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A: A[i, j] -> U[i*m+j, c]
    # Indices for B: B[j, k] -> V[j*p+k, c]
    # Indices for C: C[i, k] -> W[k*n+i, c]
    
    # Part 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 is 2x2, B1 is 2x3 (but we only use cols 0,1 for Strassen, col 2 for schoolbook)
    # Actually B1 has rows 0,1. So B1 entries are B[0,0], B[0,1], B[0,2] and B[1,0], B[1,1], B[1,2].
    # Strassen uses B[0,0], B[0,1], B[1,0], B[1,1].
    # Schoolbook uses B[0,2], B[1,2].
    
    # Strassen for A1 * B1[:, :2]
    # m1 = (A00 + A11) * (B00 + B11)
    # U[0, 0] = 1 (A00), U[3, 0] = 1 (A11) -> A00+A11
    # V[0, 0] = 1 (B00), V[4, 0] = 1 (B11) -> B00+B11
    # W[0, 0] = 1 (C00), W[3, 0] = 1 (C11) -> C00, C11
    # Wait, W index is k*n+i.
    # C00 -> k=0, i=0 -> W[0, 0]
    # C11 -> k=1, i=1 -> W[1*2+1, 0] = W[3, 0]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (A10 + A11) * B00
    U[1][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C10 -> k=0, i=1 -> W[1, 1]? No.
    # C10 is i=1, k=0. W[0*2+1, 1] = W[1, 1].
    # Let's re-verify W index. W[k*n+i].
    # C[i,k].
    # C00: i=0, k=0 -> W[0, 0].
    # C01: i=0, k=1 -> W[1, 0].
    # C02: i=0, k=2 -> W[2, 0].
    # C10: i=1, k=0 -> W[0, 1].
    # C11: i=1, k=1 -> W[1, 1].
    # C12: i=1, k=2 -> W[2, 1].
    
    # Correction:
    # m1: C00 += m1, C11 += m1
    W[0][0] = 1
    W[1][1] = 1
    
    # m2: C10 += m2
    W[0][1] = 1
    
    # m3 = A00 * (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1 # B01 - B11. B01 is V[1], B11 is V[5]?
    # V index: j*p+k.
    # B00: j=0, k=0 -> V[0]
    # B01: j=0, k=1 -> V[1]
    # B02: j=0, k=2 -> V[2]
    # B10: j=1, k=0 -> V[3]
    # B11: j=1, k=1 -> V[4]
    # B12: j=1, k=2 -> V[5]
    # B20: j=2, k=0 -> V[6]
    # B21: j=2, k=1 -> V[7]
    # B22: j=2, k=2 -> V[8]
    
    # So B01 is V[1], B11 is V[4].
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1 # C01
    
    # m4 = A11 * (B10 - B00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C10
    
    # m5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1 # C00 -= m5
    W[1][4] = 1 # C01 += m5
    
    # m6 = (A10 - A00) * (B00 + B01)
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1 # C11 += m6
    
    # m7 = (A01 - A11) * (B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1 # C00 += m7
    W[1][6] = -1 # C11 -= m7
    
    # Part 1b: A1 * B1[:, 2] (Schoolbook)
    # C02 += A00*B02 + A01*B12
    # C12 += A10*B02 + A11*B12
    
    # m8 = A00 * B02
    U[0][7] = 1
    V[2][7] = 1
    W[2][7] = 1 # C02
    
    # m9 = A01 * B12
    U[1][8] = 1
    V[5][8] = 1
    W[2][8] = 1 # C02
    
    # m10 = A10 * B02
    U[1][9] = 1 # A10 is U[1]? No.
    # A indices:
    # A00: U[0]
    # A01: U[1]
    # A02: U[2]
    # A10: U[3]
    # A11: U[4]
    # A12: U[5]
    
    # Correction on U indices:
    # U[i*m+j].
    # i=0, j=0 -> U[0]
    # i=0, j=1 -> U[1]
    # i=0, j=2 -> U[2]
    # i=1, j=0 -> U[3]
    # i=1, j=1 -> U[4]
    # i=1, j=2 -> U[5]
    
    # So A10 is U[3], A11 is U[4].
    
    # Redo Strassen with correct U indices:
    # m1: (A00+A11) -> U[0]+U[4]
    U[0][0] = 1; U[4][0] = 1
    # m2: (A10+A11) -> U[3]+U[4]
    U[3][1] = 1; U[4][1] = 1
    # m3: A00 -> U[0]
    U[0][2] = 1
    # m4: A11 -> U[4]
    U[4][3] = 1
    # m5: (A00+A01) -> U[0]+U[1]
    U[0][4] = 1; U[1][4] = 1
    # m6: (A10-A00) -> U[3]-U[0]
    U[3][5] = 1; U[0][5] = -1
    # m7: (A01-A11) -> U[1]-U[4]
    U[1][6] = 1; U[4][6] = -1
    
    # Part 1b:
    # m8: A00 * B02
    U[0][7] = 1
    V[2][7] = 1
    W[2][7] = 1
    
    # m9: A01 * B12
    U[1][8] = 1
    V[5][8] = 1
    W[2][8] = 1
    
    # m10: A10 * B02
    U[3][9] = 1
    V[2][9] = 1
    W[2][9] = 1 # C12 -> W[2, 1]? No.
    # C12: i=1, k=2 -> W[2*2+1, 9] = W[5, 9].
    W[5][9] = 1
    
    # m11: A11 * B12
    U[4][10] = 1
    V[5][10] = 1
    W[5][10] = 1 # C12
    
    # Part 2: A2 (col 2) * B2 (row 2)
    # A2: A02 (U[2]), A12 (U[5])
    # B2: B20 (V[6]), B21 (V[7]), B22 (V[8])
    
    # m12: A02 * B20
    U[2][11] = 1
    V[6][11] = 1
    W[0][11] = 1 # C00
    
    # m13: A02 * B21
    U[2][12] = 1
    V[7][12] = 1
    W[1][12] = 1 # C01
    
    # m14: A02 * B22
    U[2][13] = 1
    V[8][13] = 1
    W[2][13] = 1 # C02
    
    # m15: A12 * B20
    U[5][14] = 1
    V[6][14] = 1
    W[0][14] = 1 # C10
    
    # m16: A12 * B21
    U[5][15] = 1
    V[7][15] = 1
    W[1][15] = 1 # C11
    
    # m17: A12 * B22
    U[5][16] = 1
    V[8][16] = 1
    W[2][16] = 1 # C12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
