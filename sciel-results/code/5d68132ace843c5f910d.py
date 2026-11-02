"""
Rewritten program for matrix multiplication 2x2 by 2x4.
Uses a rank-14 decomposition based on two independent Strassen-like blocks.
"""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    # U shape (n*m, rank) = (4, 14)
    # V shape (m*p, rank) = (8, 14)
    # W shape (p*n, rank) = (8, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1 (Columns 0-6): Computes A @ B[:, 0:2]
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices: 0:e, 1:f, 2:g, 3:h (B cols 0,1)
    # W indices: 0:C00, 1:C10, 2:C01, 3:C11
    
    # Col 0: M1 = (a+d)(e+h) -> C00(+1), C11(+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Col 1: M2 = (c+d)e -> C10(+1), C11(-1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Col 2: M3 = a(f-h) -> C01(+1), C11(+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Col 3: M4 = d(g-e) -> C00(+1), C10(+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Col 4: M5 = (a+b)h -> C00(-1), C01(+1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Col 5: M6 = (c-a)(e+f) -> C11(+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Col 6: M7 = (b-d)(g+h) -> C00(+1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # Block 2 (Columns 7-13): Computes A @ B[:, 2:4]
    # V indices: 4:i, 5:j, 6:k, 7:l (B cols 2,3)
    # W indices: 4:C02, 5:C12, 6:C03, 7:C13
    
    # Col 7: M'_1 = (a+d)(i+l) -> C02(+1), C13(+1) -- Wait, check formulas.
    # Formula: C02 = M1+M4-M5+M7, C13 = M1-M2+M3+M6.
    # Let's re-verify W mapping for Block 2.
    # C02 (Row 4) coeff: M1(+1), M4(+1), M5(-1), M7(+1)
    # C12 (Row 5) coeff: M2(+1), M4(+1)
    # C03 (Row 6) coeff: M3(+1), M5(+1)
    # C13 (Row 7) coeff: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # Col 7: M'_1
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # Col 8: M'_2
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # Col 9: M'_3
    U[0][9] = 1
    V[5][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # Col 10: M'_4
    U[3][10] = 1
    V[6][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # Col 11: M'_5
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # Col 12: M'_6
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[7][12] = 1
    
    # Col 13: M'_7
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
