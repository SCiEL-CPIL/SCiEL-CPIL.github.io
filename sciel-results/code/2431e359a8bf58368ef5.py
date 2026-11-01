"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-14 decomposition using Strassen for each 2x2 half of the 2x4 matrix
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14

    # Strassen products for 2x2 * 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For left half (k=0,1): B indices j*4+k
    # For right half (k=2,3): B indices j*4+k

    # Define Strassen factor patterns for a 2x2 multiplication
    # Each product r has (U_coeffs, V_coeffs, W_coeffs)
    # U is over (i,j) pairs: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V is over (j,k) pairs
    # W is over (k,i) pairs

    # Left half: k in {0,1}
    # B_left[0,0] = V[0*4+0] = V[0]
    # B_left[0,1] = V[0*4+1] = V[1]
    # B_left[1,0] = V[1*4+0] = V[4]
    # B_left[1,1] = V[1*4+1] = V[5]
    # C_left[0,0] = W[0*2+0] = W[0]
    # C_left[0,1] = W[1*2+0] = W[2]
    # C_left[1,0] = W[0*2+1] = W[1]
    # C_left[1,1] = W[1*2+1] = W[3]

    # Right half: k in {2,3}
    # B_right[0,0] = V[0*4+2] = V[2]
    # B_right[0,1] = V[0*4+3] = V[3]
    # B_right[1,0] = V[1*4+2] = V[6]
    # B_right[1,1] = V[1*4+3] = V[7]
    # C_right[0,0] = W[2*2+0] = W[4]
    # C_right[0,1] = W[3*2+0] = W[6]
    # C_right[1,0] = W[2*2+1] = W[5]
    # C_right[1,1] = W[3*2+1] = W[7]

    # Strassen products for left half (r=0..6)
    # r=0: M1 = (A00+A11)(B00+B10) [using B10=B_left[1,0]=V[4], B11=B_left[1,1]=V[5]]
    # Wait, let me be more careful with Strassen for 2x2*2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    # For left half:
    # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
    # e=B_left[0,0]=V[0], f=B_left[0,1]=V[1], g=B_left[1,0]=V[4], h=B_left[1,1]=V[5]

    # r=0: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = 1; W[3][0] = 1  # M1 contributes to all C entries

    # r=1: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1  # C[1,0]=M2+..., C[1,1]=-M2+...

    # r=2: M3 = a(g-h)
    U[0][2] = 1
    V[4][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1  # C[0,1]=M3+..., C[1,1]=+M3+...

    # r=3: M4 = d(f-g)
    U[3][3] = 1
    V[1][3] = 1; V[4][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C[0,0]=+M4+..., C[1,0]=+M4+...

    # r=4: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C[0,0]=-M5+..., C[0,1]=+M5+...

    # r=5: M6 = (c-a)(e+g)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[4][5] = 1
    W[3][5] = 1  # C[1,1]=+M6+...

    # r=6: M7 = (b-d)(f+h)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[5][6] = 1
    W[0][6] = 1  # C[0,0]=+M7+...

    # For right half: k in {2,3}
    # e=B_right[0,0]=V[2], f=B_right[0,1]=V[3], g=B_right[1,0]=V[6], h=B_right[1,1]=V[7]
    # C_right[0,0]=W[4], C_right[0,1]=W[6], C_right[1,0]=W[5], C_right[1,1]=W[7]

    # r=7: M1' = (a+d)(e'+h')
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[6][7] = 1; W[5][7] = 1; W[7][7] = 1

    # r=8: M2' = (c+d)e'
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1

    # r=9: M3' = a(g'-h')
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1

    # r=10: M4' = d(f'-g')
    U[3][10] = 1
    V[3][10] = 1; V[6][10] = -1
    W[4][10] = 1; W[5][10] = 1

    # r=11: M5' = (a+b)h'
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1

    # r=12: M6' = (c-a)(e'+g')
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[6][12] = 1
    W[7][12] = 1

    # r=13: M7' = (b-d)(f'+h')
    U[1][13] = 1; U[3][13] = -1
    V[3][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
