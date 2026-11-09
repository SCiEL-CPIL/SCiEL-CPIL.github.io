"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # --- Block 1: M_{2,3,2} (Rank 11) ---
    # B uses columns 0, 1 (k=0,1)
    # A uses all rows/cols (i=0,1; j=0,1,2)
    # C uses columns 0, 1 (k=0,1)
    
    # Sub-block 1.1: M_{2,2,2} (Strassen)
    # A_left (j=0,1), B_top (j=0,1), C (i=0,1; k=0,1)
    
    # M1 = (a00+a11)(b00+b11)
    U[0][col_offset] += 1; U[3][col_offset] += 1
    V[0][col_offset] += 1; V[1][col_offset] += 1
    W[0][col_offset] += 1  # C[0,0]
    W[5][col_offset] += 1  # C[1,1] (index k*n+i = 1*2+1=3? No. W[k*n+i]. k=1, i=1 -> 1*2+1=3. Wait.
    # W index: k*n + i.
    # k=0, i=0 -> 0.
    # k=0, i=1 -> 1.
    # k=1, i=0 -> 2.
    # k=1, i=1 -> 3.
    # So W[3] is C[1,1].
    W[3][col_offset] += 1
    col_offset += 1

    # M2 = (a10+a11)b00
    U[3][col_offset] += 1; U[4][col_offset] += 1  # a10 (idx 1*3+0=3), a11 (idx 1*3+1=4)
    V[0][col_offset] += 1
    W[2][col_offset] += 1  # C[1,0] (k=1, i=0 -> 2)
    W[3][col_offset] -= 1  # C[1,1] (k=1, i=1 -> 3)
    col_offset += 1

    # M3 = a00(b01-b11)
    U[0][col_offset] += 1
    V[1][col_offset] += 1; V[2][col_offset] -= 1  # b01 (idx 0*5+1=1), b11 (idx 1*5+1=6? No. B is 3x5.
    # V index j*p+k.
    # Block 1 uses k=0,1.
    # B_top is j=0,1.
    # b00: j=0, k=0 -> 0.
    # b01: j=0, k=1 -> 1.
    # b10: j=1, k=0 -> 5.
    # b11: j=1, k=1 -> 6.
    # Correction:
    V[1][col_offset] += 1; V[6][col_offset] -= 1
    W[1][col_offset] += 1  # C[0,1] (k=1, i=0 -> 1)
    W[3][col_offset] += 1  # C[1,1] (k=1, i=1 -> 3)
    col_offset += 1

    # M4 = a11(b10-b00)
    U[4][col_offset] += 1
    V[5][col_offset] += 1; V[0][col_offset] -= 1
    W[0][col_offset] += 1  # C[0,0]
    W[2][col_offset] += 1  # C[1,0]
    col_offset += 1

    # M5 = (a00+a01)b11
    U[0][col_offset] += 1; U[1][col_offset] += 1
    V[6][col_offset] += 1
    W[0][col_offset] -= 1  # C[0,0]
    W[1][col_offset] += 1  # C[0,1]
    col_offset += 1

    # M6 = (a10-a00)(b00+b01)
    U[3][col_offset] += 1; U[0][col_offset] -= 1
    V[0][col_offset] += 1; V[1][col_offset] += 1
    W[3][col_offset] += 1  # C[1,1]
    col_offset += 1

    # M7 = (a01-a11)(b10+b11)
    U[1][col_offset] += 1; U[4][col_offset] -= 1
    V[5][col_offset] += 1; V[6][col_offset] += 1
    W[0][col_offset] += 1  # C[0,0]
    col_offset += 1

    # Sub-block 1.2: M_{2,1,2} (Schoolbook)
    # a_right (j=2), b_bot (j=2)
    # a02: U[2], a12: U[5]
    # b20: V[10], b21: V[11]
    
    # N1 = a02 * b20 -> C[0,0]
    U[2][col_offset] += 1
    V[10][col_offset] += 1
    W[0][col_offset] += 1
    col_offset += 1

    # N2 = a02 * b21 -> C[0,1]
    U[2][col_offset] += 1
    V[11][col_offset] += 1
    W[1][col_offset] += 1
    col_offset += 1

    # N3 = a12 * b20 -> C[1,0]
    U[5][col_offset] += 1
    V[10][col_offset] += 1
    W[2][col_offset] += 1
    col_offset += 1

    # N4 = a12 * b21 -> C[1,1]
    U[5][col_offset] += 1
    V[11][col_offset] += 1
    W[3][col_offset] += 1
    col_offset += 1
    
    # --- Block 2: M_{2,3,3} (Rank 17) ---
    # B uses columns 2,3,4 (k=2,3,4)
    # A uses all rows/cols
    # C uses columns 2,3,4

    # Sub-block 2.1: M_{2,2,2} (Strassen)
    # A_left (j=0,1), B_top_left (j=0,1; k=2,3)
    
    # M1' = (a00+a11)(b02+b13)
    U[0][col_offset] += 1; U[3][col_offset] += 1
    V[2][col_offset] += 1; V[7][col_offset] += 1  # b02: 0*5+2=2, b13: 1*5+3=8? No.
    # B index j*p+k.
    # b02: j=0, k=2 -> 2.
    # b13: j=1, k=3 -> 1*5+3=8.
    # Wait, Strassen uses 2x2 block.
    # B_top is 2x3. Left 2x2 is cols 2,3.
    # b02 (0,2), b03 (0,3), b12 (1,2), b13 (1,3).
    # M1' uses b02 and b13? No, diagonal.
    # (b02 + b13) is not standard Strassen.
    # Standard Strassen on B = [[b02, b03], [b12, b13]].
    # M1' = (a00+a11)(b02+b13).
    V[2][col_offset] += 1; V[8][col_offset] += 1
    W[4][col_offset] += 1  # C[0,2] (k=2, i=0 -> 2*2+0=4)
    W[9][col_offset] += 1  # C[1,3] (k=3, i=1 -> 3*2+1=7? No. k=3, i=1 -> 3*2+1=7.
    # W index k*n+i.
    # k=2, i=0 -> 4.
    # k=3, i=1 -> 7.
    # So W[7].
    W[7][col_offset] += 1
    col_offset += 1

    # M2' = (a10+a11)b02
    U[3][col_offset] += 1; U[4][col_offset] += 1
    V[2][col_offset] += 1
    W[6][col_offset] += 1  # C[1,2] (k=2, i=1 -> 5? No. k=2, i=1 -> 2*2+1=5.
    # W[5].
    W[5][col_offset] += 1
    W[7][col_offset] -= 1
    col_offset += 1

    # M3' = a00(b03-b13)
    U[0][col_offset] += 1
    V[3][col_offset] += 1; V[8][col_offset] -= 1
    W[5][col_offset] += 1  # C[0,3] (k=3, i=0 -> 6? No. k=3, i=0 -> 3*2+0=6.
    # W[6].
    W[6][col_offset] += 1
    W[7][col_offset] += 1
    col_offset += 1

    # M4' = a11(b12-b02)
    U[4][col_offset] += 1
    V[7][col_offset] += 1; V[2][col_offset] -= 1
    W[4][col_offset] += 1
    W[5][col_offset] += 1
    col_offset += 1

    # M5' = (a00+a01)b13
    U[0][col_offset] += 1; U[1][col_offset] += 1
    V[8][col_offset] += 1
    W[4][col_offset] -= 1
    W[6][col_offset] += 1
    col_offset += 1

    # M6' = (a10-a00)(b02+b03)
    U[3][col_offset] += 1; U[0][col_offset] -= 1
    V[2][col_offset] += 1; V[3][col_offset] += 1
    W[7][col_offset] += 1
    col_offset += 1

    # M7' = (a01-a11)(b12+b13)
    U[1][col_offset] += 1; U[4][col_offset] -= 1
    V[7][col_offset] += 1; V[8][col_offset] += 1
    W[4][col_offset] += 1
    col_offset += 1

    # Sub-block 2.2: M_{2,2,1} (Schoolbook)
    # A_left (j=0,1), b_top_right (j=0,1; k=4)
    # b04: V[4], b14: V[9]
    
    # P1 = a00 * b04 -> C[0,4]
    U[0][col_offset] += 1
    V[4][col_offset] += 1
    W[8][col_offset] += 1  # C[0,4] (k=4, i=0 -> 8)
    col_offset += 1

    # P2 = a01 * b14 -> C[0,4]
    U[1][col_offset] += 1
    V[9][col_offset] += 1
    W[8][col_offset] += 1
    col_offset += 1

    # P3 = a10 * b04 -> C[1,4]
    U[3][col_offset] += 1
    V[4][col_offset] += 1
    W[9][col_offset] += 1  # C[1,4] (k=4, i=1 -> 9)
    col_offset += 1

    # P4 = a11 * b14 -> C[1,4]
    U[4][col_offset] += 1
    V[9][col_offset] += 1
    W[9][col_offset] += 1
    col_offset += 1

    # Sub-block 2.3: M_{2,1,3} (Schoolbook)
    # a_right (j=2), b_bot (j=2; k=2,3,4)
    # b22: V[12], b23: V[13], b24: V[14]
    
    # Q1 = a02 * b22 -> C[0,2]
    U[2][col_offset] += 1
    V[12][col_offset] += 1
    W[4][col_offset] += 1
    col_offset += 1

    # Q2 = a02 * b23 -> C[0,3]
    U[2][col_offset] += 1
    V[13][col_offset] += 1
    W[6][col_offset] += 1
    col_offset += 1

    # Q3 = a02 * b24 -> C[0,4]
    U[2][col_offset] += 1
    V[14][col_offset] += 1
    W[8][col_offset] += 1
    col_offset += 1

    # Q4 = a12 * b22 -> C[1,2]
    U[5][col_offset] += 1
    V[12][col_offset] += 1
    W[5][col_offset] += 1
    col_offset += 1

    # Q5 = a12 * b23 -> C[1,3]
    U[5][col_offset] += 1
    V[13][col_offset] += 1
    W[7][col_offset] += 1
    col_offset += 1

    # Q6 = a12 * b24 -> C[1,4]
    U[5][col_offset] += 1
    V[14][col_offset] += 1
    W[9][col_offset] += 1
    col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
