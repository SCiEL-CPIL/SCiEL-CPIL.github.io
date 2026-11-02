"""
Rewritten program for matrix multiplication 2x3 by 3x5.
Uses a block decomposition strategy:
1. Split A into A_left (2x2) and A_right (2x1).
2. Split B into B_top (2x5) and B_bottom (1x5).
3. C = A_left @ B_top + A_right @ B_bottom.
4. Use Strassen algorithm (Rank 7) for the 2x2 by 2x2 part of B_top (first 2 cols).
5. Use Schoolbook (Rank 8) for the 2x2 by 2x3 part of B_top (last 3 cols).
6. Use Schoolbook (Rank 10) for the 2x1 by 1x5 part (A_right * B_bottom).
Total Rank = 7 + 8 + 10 = 25.
"""
DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # We will build U (6 x Rank), V (15 x Rank), W (10 x Rank)
    # A indices: 0,1,2 (row 0), 3,4,5 (row 1)
    # B indices: 0..4 (row 0), 5..9 (row 1), 10..14 (row 2)
    # C indices: 0..4 (row 0), 5..9 (row 1) -> Index k*n+i
    
    # Block 1: A_left (cols 0,1 of A) * B_top_left (cols 0,1 of B_top -> rows 0,1 of B, cols 0,1 of C)
    # A_left uses A indices 0, 1, 3, 4.
    # B_top_left uses B indices 0, 1, 5, 6.
    # Output C indices: 0,1 (row 0, col 0,1) and 5,6 (row 1, col 0,1).
    
    # Strassen Decomposition for M_{2,2,2}
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    U_strassen = []
    V_strassen = []
    W_strassen = []
    
    # M1
    # A: a00 (idx 0), a11 (idx 4) -> Wait, A_left is A[:, 0:2].
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1).
    # B indices: 0(0,0), 1(0,1), 5(1,0), 6(1,1).
    # C indices: 0(0,0), 1(0,1), 5(1,0), 6(1,1).
    
    # M1
    U_strassen.append([1, 0, 0, 1, 0, 0]) # a00 + a11
    V_strassen.append([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # b00 + b11 (b00 idx 0, b11 idx 6)
    # Correction: V must be length 15.
    # b00 is idx 0. b11 is idx 6 (Row 1 Col 1 of B? B is 3x5.
    # B_top is rows 0,1.
    # B indices:
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # B_top_left: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6.
    
    # Let's rebuild V properly for Strassen
    # V must be size 15.
    # M1: b00 + b11 -> idx 0 and 6.
    V_strassen = []
    U_strassen = []
    W_strassen = []
    
    def add_term(U_coeffs, V_coeffs, W_coeffs):
        U_strassen.append(U_coeffs)
        V_strassen.append(V_coeffs)
        W_strassen.append(W_coeffs)
        
    # M1
    # U: a00 (0), a11 (4). Coeff 1,1.
    U1 = [0]*6; U1[0] = 1; U1[4] = 1
    # V: b00 (0), b11 (6). Coeff 1,1.
    V1 = [0]*15; V1[0] = 1; V1[6] = 1
    # W: c00 (0), c11 (6).
    # c00 gets +1. c11 gets +1.
    W1 = [0]*10; W1[0] = 1; W1[6] = 1
    add_term(U1, V1, W1)
    
    # M2
    # U: a10 (3), a11 (4). Coeff 1,1.
    U2 = [0]*6; U2[3] = 1; U2[4] = 1
    # V: b00 (0). Coeff 1.
    V2 = [0]*15; V2[0] = 1
    # W: c10 (5). Coeff 1.
    W2 = [0]*10; W2[5] = 1
    add_term(U2, V2, W2)
    
    # M3
    # U: a00 (0). Coeff 1.
    U3 = [0]*6; U3[0] = 1
    # V: b01 (1) - b11 (6). Coeff 1, -1.
    V3 = [0]*15; V3[1] = 1; V3[6] = -1
    # W: c01 (1). Coeff 1.
    W3 = [0]*10; W3[1] = 1
    add_term(U3, V3, W3)
    
    # M4
    # U: a11 (4). Coeff 1.
    U4 = [0]*6; U4[4] = 1
    # V: b10 (5) - b00 (0). Coeff 1, -1.
    V4 = [0]*15; V4[5] = 1; V4[0] = -1
    # W: c00 (0), c10 (5).
    # c00 gets +1. c10 gets +1.
    W4 = [0]*10; W4[0] = 1; W4[5] = 1
    add_term(U4, V4, W4)
    
    # M5
    # U: a00 (0), a01 (1). Coeff 1,1.
    U5 = [0]*6; U5[0] = 1; U5[1] = 1
    # V: b11 (6). Coeff 1.
    V5 = [0]*15; V5[6] = 1
    # W: c00 (0), c01 (1).
    # c00 gets -1. c01 gets +1.
    W5 = [0]*10; W5[0] = -1; W5[1] = 1
    add_term(U5, V5, W5)
    
    # M6
    # U: a10 (3) - a00 (0). Coeff 1, -1.
    U6 = [0]*6; U6[3] = 1; U6[0] = -1
    # V: b00 (0) + b01 (1). Coeff 1,1.
    V6 = [0]*15; V6[0] = 1; V6[1] = 1
    # W: c11 (6). Coeff 1.
    W6 = [0]*10; W6[6] = 1
    add_term(U6, V6, W6)
    
    # M7
    # U: a01 (1) - a11 (4). Coeff 1, -1.
    U7 = [0]*6; U7[1] = 1; U7[4] = -1
    # V: b10 (5) + b11 (6). Coeff 1,1.
    V7 = [0]*15; V7[5] = 1; V7[6] = 1
    # W: c00 (0). Coeff 1.
    W7 = [0]*10; W7[0] = 1
    add_term(U7, V7, W7)

    # Block 2: A_left * B_top_right (cols 2,3,4 of B)
    # Schoolbook for 2x2 by 2x3. Rank 4*3 = 12.
    # A indices: 0,1,3,4.
    # B indices: 2,3,4 (row 0), 7,8,9 (row 1).
    # C indices: 2,3,4 (row 0), 7,8,9 (row 1).
    
    U_school = []
    V_school = []
    W_school = []
    
    for k in [2,3,4]: # col of C
        # Term for A00 * B_k0
        U_val = [0]*6; U_val[0] = 1
        V_val = [0]*15; V_val[k] = 1
        W_val = [0]*10; W_val[k] = 1 # c_0k
        U_school.append(U_val); V_school.append(V_val); W_school.append(W_val)
        
        # Term for A01 * B_k1
        U_val = [0]*6; U_val[1] = 1
        V_val = [0]*15; V_val[5+k] = 1
        W_val = [0]*10; W_val[k] = 1 # c_0k
        U_school.append(U_val); V_school.append(V_val); W_school.append(W_val)
        
        # Term for A10 * B_k0
        U_val = [0]*6; U_val[3] = 1
        V_val = [0]*15; V_val[k] = 1
        W_val = [0]*10; W_val[5+k] = 1 # c_1k
        U_school.append(U_val); V_school.append(V_val); W_school.append(W_val)
        
        # Term for A11 * B_k1
        U_val = [0]*6; U_val[4] = 1
        V_val = [0]*15; V_val[5+k] = 1
        W_val = [0]*10; W_val[5+k] = 1 # c_1k
        U_school.append(U_val); V_school.append(V_val); W_school.append(W_val)

    # Block 3: A_right * B_bottom
    # A_right: col 2 of A. Indices 2 (a02), 5 (a12).
    # B_bottom: row 2 of B. Indices 10,11,12,13,14.
    # C: all columns.
    # Rank 2*5 = 10.
    
    U_rem = []
    V_rem = []
    W_rem = []
    
    for k in [0,1,2,3,4]: # col of C
        # A02 * B_2k
        U_val = [0]*6; U_val[2] = 1
        V_val = [0]*15; V_val[10+k] = 1
        W_val = [0]*10; W_val[k] = 1 # c_0k
        U_rem.append(U_val); V_rem.append(V_val); W_rem.append(W_val)
        
        # A12 * B_2k
        U_val = [0]*6; U_val[5] = 1
        V_val = [0]*15; V_val[10+k] = 1
        W_val = [0]*10; W_val[5+k] = 1 # c_1k
        U_rem.append(U_val); V_rem.append(V_val); W_rem.append(W_val)

    # Combine all blocks
    U_total = U_strassen + U_school + U_rem
    V_total = V_strassen + V_school + V_rem
    W_total = W_strassen + W_school + W_rem
    
    return {"U": U_total, "V": V_total, "W": W_total}

# EVOLVE-BLOCK-END
