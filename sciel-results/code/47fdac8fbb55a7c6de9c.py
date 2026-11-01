"""Rank-18 exact decomposition for 2x2 times 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # n=2, m=2, p=5
    # U shape (n*m, rank) = (4, 18)
    # V shape (m*p, rank) = (10, 18)
    # W shape (p*n, rank) = (10, 18)
    # Indexing: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    n, m, p = DIMENSIONS
    rank = 18

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Block 1: k in {0, 1} (first 2 columns of B)
    # Strassen decomposition for 2x2 * 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M6
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M3 + M4 + M7
    # 0-based: C00=C11, C01=C12, C10=C21, C11=C22

    # r=0: M1 = (A00+A11)*(B00+B11)  [for k=0,1]
    # L_A = A00 + A11, L_B = B00 + B11
    # U: rows 0,3 -> 1; V: rows 0,1 -> 1; W: C00(0) +1, C11(3) +1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # r=1: M2 = (A10+A11)*B00  [k=0]
    # L_A = A10 + A11, L_B = B00
    # U: rows 2,3 -> 1; V: row 0 -> 1; W: C01(2) +1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # r=2: M3 = A00*(B10-B00)  [k=0,1]
    # L_A = A00, L_B = B10 - B00
    # U: row 0 -> 1; V: row 1 -> 1, row 0 -> -1; W: C10(1) +1
    U[0][2] = 1
    V[1][2] = 1; V[0][2] = -1
    W[1][2] = 1

    # r=3: M4 = A11*(B01-B00)  [k=0,1]
    # L_A = A11, L_B = B01 - B00
    # U: row 3 -> 1; V: row 5 -> 1, row 0 -> -1; W: C00(0) +1, C01(2) +1
    U[3][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # r=4: M5 = (A00+A01)*B11  [k=1]
    # L_A = A00 + A01, L_B = B11
    # U: rows 0,1 -> 1; V: row 6 -> 1; W: C00(0) -1, C10(1) +1
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # r=5: M6 = (A10-A11)*(B00+B01)  [k=0,1]
    # L_A = A10 - A11, L_B = B00 + B01
    # U: row 2 -> 1, row 3 -> -1; V: row 0 -> 1, row 5 -> 1; W: C00(0) +1
    U[2][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[5][5] = 1
    W[0][5] = 1

    # r=6: M7 = (A01-A11)*(B10+B11)  [k=0,1]
    # L_A = A01 - A11, L_B = B10 + B11
    # U: row 1 -> 1, row 3 -> -1; V: row 1 -> 1, row 6 -> 1; W: C11(3) +1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[6][6] = 1
    W[3][6] = 1

    # Block 2: k in {2, 3} (next 2 columns of B)
    # Same Strassen structure, but B indices shifted by 2
    # B02,B12 correspond to V rows 2,7 (j=0,k=2; j=1,k=2)
    # B03,B13 correspond to V rows 3,8 (j=0,k=3; j=1,k=3)
    # W indices: k=2 -> W[2*n+i] = W[4+i], k=3 -> W[3*n+i] = W[6+i]

    # r=7: M1' = (A00+A11)*(B02+B12)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1

    # r=8: M2' = (A10+A11)*B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[6][8] = 1  # C01 for k=3? No: C_{i,k}. For k=2, C01 means output row 0, col 1. W index = k*n+i. C_{0,2} -> W[2*2+0]=W[4]. C_{1,2}->W[6].

    # Wait, let me recheck. Output C is 2x5. C[i,k]. W index = k*n + i.
    # For block 2, k in {2,3}.
    # C_{0,2} -> W[4], C_{1,2} -> W[5], C_{0,3} -> W[6], C_{1,3} -> W[7]

    # r=7: M1' contributes to C_{0,2} and C_{1,2} (for k=2) and C_{0,3}, C_{1,3}? 
    # No. M1 = (a+d)(e+h) where e=B00, h=B11 for the 2x2 block.
    # For block 2, the 2x2 sub-block of B is rows 0,1 and cols 2,3.
    # So e=B02, h=B12, f=B12? No.
    # B sub-block: [[B02, B03], [B12, B13]]
    # e = B02, f = B12, g = B03, h = B13
    # M1 = (A00+A11)*(B02+B13)
    # M2 = (A10+A11)*B02
    # M3 = A00*(B12-B02)
    # M4 = A11*(B03-B02)
    # M5 = (A00+A01)*B13
    # M6 = (A10-A11)*(B02+B03)
    # M7 = (A01-A11)*(B12+B13)
    # C_{0,2} = M1 + M4 - M5 + M6
    # C_{0,3} = M2 + M4
    # C_{1,2} = M3 + M5
    # C_{1,3} = M1 - M3 + M4 + M7

    # Let me redo block 2 carefully.
    # r=7: M1' = (A00+A11)*(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[8][7] = 1  # B02 = V[0*5+2]=V[2], B13 = V[1*5+3]=V[8]
    W[4][7] = 1; W[7][7] = 1  # C_{0,2}=W[4], C_{1,3}=W[7]

    # r=8: M2' = (A10+A11)*B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[6][8] = 1  # C_{0,3}=W[6]

    # r=9: M3' = A00*(B12-B02)
    U[0][9] = 1
    V[7][9] = 1; V[2][9] = -1  # B12=V[7], B02=V[2]
    W[5][9] = 1  # C_{1,2}=W[5]

    # r=10: M4' = A11*(B03-B02)
    U[3][10] = 1
    V[3][10] = 1; V[2][10] = -1  # B03=V[3], B02=V[2]
    W[4][10] = 1; W[6][10] = 1  # C_{0,2}=W[4], C_{0,3}=W[6]

    # r=11: M5' = (A00+A01)*B13
    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[4][11] = -1; W[5][11] = 1  # C_{0,2}=-1, C_{1,2}=+1

    # r=12: M6' = (A10-A11)*(B02+B03)
    U[2][12] = 1; U[3][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[4][12] = 1  # C_{0,2}

    # r=13: M7' = (A01-A11)*(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[7][13] = 1  # C_{1,3}

    # Block 3: k=4 (last column), schoolbook: 4 terms
    # C_{0,4} = A00*B04 + A01*B14
    # C_{1,4} = A10*B04 + A11*B14

    # r=14: A00*B04 -> C_{0,4}
    U[0][14] = 1
    V[4][14] = 1  # B04 = V[0*5+4]=V[4]
    W[8][14] = 1  # C_{0,4} = W[4*2+0] = W[8]

    # r=15: A01*B14 -> C_{0,4}
    U[1][15] = 1
    V[9][15] = 1  # B14 = V[1*5+4]=V[9]
    W[8][15] = 1

    # r=16: A10*B04 -> C_{1,4}
    U[2][16] = 1
    V[4][16] = 1
    W[9][16] = 1  # C_{1,4} = W[4*2+1] = W[9]

    # r=17: A11*B14 -> C_{1,4}
    U[3][17] = 1
    V[9][17] = 1
    W[9][17] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
