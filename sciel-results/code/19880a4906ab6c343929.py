import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    
    # We split A (2x5) into A_L (2x2), A_M (2x1), A_R (2x2)
    # We split B (5x5) into B_T (2x5), B_M (1x5), B_B (2x5)
    
    # C = A_L B_T + A_M B_M + A_R B_B
    
    # B_T is 2x5. Split into BT_1 (2x2), BT_2 (2x2), BT_3 (2x1)
    # B_B is 2x5. Split into BB_1 (2x2), BB_2 (2x2), BB_3 (2x1)
    
    # So we need:
    # 1. A_L * BT_1 (2x2 * 2x2) -> Rank 7
    # 2. A_L * BT_2 (2x2 * 2x2) -> Rank 7
    # 3. A_L * BT_3 (2x2 * 2x1) -> Rank 4
    # 4. A_R * BB_1 (2x2 * 2x2) -> Rank 7
    # 5. A_R * BB_2 (2x2 * 2x2) -> Rank 7
    # 6. A_R * BB_3 (2x2 * 2x1) -> Rank 4
    # 7. A_M * B_M (2x1 * 1x5) -> Rank 10
    
    # Total Rank = 7+7+4+7+7+4+10 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)] # 10 x 46
    V = [[0] * rank for _ in range(m * p)] # 25 x 46
    W = [[0] * rank for _ in range(p * n)] # 10 x 46
    
    # Helper to add a term
    # u_idx, v_idx, w_idx, coeff
    # U[u_idx][col] += coeff
    # V[v_idx][col] += coeff
    # W[w_idx][col] += coeff
    
    def add_term(u_idx, v_idx, w_idx, col, coeff=1):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    def add_strassen(A_base, B_base, C_base, col_start):
        # A is 2x2 at indices A_base + [0,1,2,3] in flattened U? 
        # U index for A[i,j] is i*m + j. Here m=2 for the sub-block.
        # But U is global 2x5. 
        # Let's map local indices to global.
        
        # A_L is cols 0,1 of A. 
        # A[i, j_local] -> U[i*5 + j_local]
        # B_T is rows 0,1 of B.
        # B[j_local, k] -> V[j_local*5 + k]
        # C is rows 0,1 of C? No, C is 2x5.
        # C[i, k] -> W[k*2 + i]
        
        # Strassen M1..M7
        # M1 = (A00+A11)(B00+B11)
        # A00: U[0], A11: U[1]
        # B00: V[0], B11: V[1]
        
        # We need to construct the linear combinations in U and V.
        # The tensor decomposition is sum of rank-1 terms.
        # We can't just add coefficients to U and V if they are shared across terms in a way that breaks rank-1.
        # Strassen terms are products of linear combinations.
        # (A00+A11) is a linear combination of A entries.
        # We need to express (A00+A11) * (B00+B11) as sum of rank 1 terms?
        # No, Strassen algorithm *is* a rank 7 decomposition.
        # The rank 1 terms are:
        # T1: (A00+A11) * (B00+B11)
        # T2: (A10+A11) * B00
        # T3: A00 * (B10-B11)
        # T4: A11 * (B01-B00)
        # T5: (A00+A01) * B11
        # T6: (A10-A00) * (B00+B01)
        # T7: (A01-A11) * (B10+B11)
        
        # Each Ti contributes to C.
        # C00 = T1 + T4 - T5 + T7
        # C01 = T3 + T5
        # C10 = T2 + T4
        # C11 = T1 - T2 + T3 + T6
        
        # We add 7 columns.
        # Col 0: T1
        # U: A00, A11 have coeff 1.
        # V: B00, B11 have coeff 1.
        # W: C00 coeff 1, C11 coeff 1.
        
        # Let's implement a generic add_strassen for 2x2x2.
        # a_indices: [u00, u10, u01, u11]
        # b_indices: [v00, v10, v01, v11]
        # c_indices: [w00, w10, w01, w11] -> Note W is [k*n+i, r]. 
        # C is 2x2. C[i,k] -> W[k*2+i].
        # C00 -> W[0], C10 -> W[1] (k=0, i=1? No, k is row of B, i is row of A. 
        # C[i,k]. W[k*n+i]. n=2.
        # C[0,0] -> W[0*2+0] = W[0]
        # C[1,0] -> W[0*2+1] = W[1]
        # C[0,1] -> W[1*2+0] = W[2]
        # C[1,1] -> W[1*2+1] = W[3]
        
        a00, a10, a01, a11 = a_indices
        b00, b10, b01, b11 = b_indices
        c00, c10, c01, c11 = c_indices
        
        # T1: (a00+a11)(b00+b11)
        # U: a00=1, a11=1. V: b00=1, b11=1.
        # W: c00=1, c11=1.
        U[a00][col_start] = 1
        U[a11][col_start] = 1
        V[b00][col_start] = 1
        V[b11][col_start] = 1
        W[c00][col_start] = 1
        W[c11][col_start] = 1
        
        # T2: (a10+a11)b00
        # U: a10=1, a11=1. V: b00=1.
        # W: c10=1, c11=-1.
        U[a10][col_start+1] = 1
        U[a11][col_start+1] = 1
        V[b00][col_start+1] = 1
        W[c10][col_start+1] = 1
        W[c11][col_start+1] = -1
        
        # T3: a00(b10-b11)
        # U: a00=1. V: b10=1, b11=-1.
        # W: c01=1, c11=1.
        U[a00][col_start+2] = 1
        V[b10][col_start+2] = 1
        V[b11][col_start+2] = -1
        W[c01][col_start+2] = 1
        W[c11][col_start+2] = 1
        
        # T4: a11(b01-b00)
        # U: a11=1. V: b01=1, b00=-1.
        # W: c00=1, c10=1.
        U[a11][col_start+3] = 1
        V[b01][col_start+3] = 1
        V[b00][col_start+3] = -1
        W[c00][col_start+3] = 1
        W[c10][col_start+3] = 1
        
        # T5: (a00+a01)b11
        # U: a00=1, a01=1. V: b11=1.
        # W: c00=-1, c01=1.
        U[a00][col_start+4] = 1
        U[a01][col_start+4] = 1
        V[b11][col_start+4] = 1
        W[c00][col_start+4] = -1
        W[c01][col_start+4] = 1
        
        # T6: (a10-a00)(b00+b01)
        # U: a10=1, a00=-1. V: b00=1, b01=1.
        # W: c11=1.
        U[a10][col_start+5] = 1
        U[a00][col_start+5] = -1
        V[b00][col_start+5] = 1
        V[b01][col_start+5] = 1
        W[c11][col_start+5] = 1
        
        # T7: (a01-a11)(b10+b11)
        # U: a01=1, a11=-1. V: b10=1, b11=1.
        # W: c00=1.
        U[a01][col_start+6] = 1
        U[a11][col_start+6] = -1
        V[b10][col_start+6] = 1
        V[b11][col_start+6] = 1
        W[c00][col_start+6] = 1

    def add_2x2x1(A_base, B_base, C_base, col_start):
        # A 2x2, B 2x1, C 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Rank 4.
        a00, a10, a01, a11 = A_base
        b0, b1 = B_base
        c0, c1 = C_base
        
        # T1: A00 B0 -> C0
        U[a00][col_start] = 1
        V[b0][col_start] = 1
        W[c0][col_start] = 1
        
        # T2: A01 B1 -> C0
        U[a01][col_start+1] = 1
        V[b1][col_start+1] = 1
        W[c0][col_start+1] = 1
        
        # T3: A10 B0 -> C1
        U[a10][col_start+2] = 1
        V[b0][col_start+2] = 1
        W[c1][col_start+2] = 1
        
        # T4: A11 B1 -> C1
        U[a11][col_start+3] = 1
        V[b1][col_start+3] = 1
        W[c1][col_start+3] = 1

    def add_2x1x5(A_base, B_base, C_base, col_start):
        # A 2x1, B 1x5, C 2x5
        # C[i,k] = A[i,0] B[0,k]
        # Rank 10.
        a0, a1 = A_base
        b0, b1, b2, b3, b4 = B_base
        c00, c10, c01, c11, c02, c12, c03, c13, c04, c14 = C_base
        
        # A0 * B0 -> C00
        U[a0][col_start] = 1
        V[b0][col_start] = 1
        W[c00][col_start] = 1
        
        # A0 * B1 -> C01
        U[a0][col_start+1] = 1
        V[b1][col_start+1] = 1
        W[c01][col_start+1] = 1
        
        # A0 * B2 -> C02
        U[a0][col_start+2] = 1
        V[b2][col_start+2] = 1
        W[c02][col_start+2] = 1
        
        # A0 * B3 -> C03
        U[a0][col_start+3] = 1
        V[b3][col_start+3] = 1
        W[c03][col_start+3] = 1
        
        # A0 * B4 -> C04
        U[a0][col_start+4] = 1
        V[b4][col_start+4] = 1
        W[c04][col_start+4] = 1
        
        # A1 * B0 -> C10
        U[a1][col_start+5] = 1
        V[b0][col_start+5] = 1
        W[c10][col_start+5] = 1
        
        # A1 * B1 -> C11
        U[a1][col_start+6] = 1
        V[b1][col_start+6] = 1
        W[c11][col_start+6] = 1
        
        # A1 * B2 -> C12
        U[a1][col_start+7] = 1
        V[b2][col_start+7] = 1
        W[c12][col_start+7] = 1
        
        # A1 * B3 -> C13
        U[a1][col_start+8] = 1
        V[b3][col_start+8] = 1
        W[c13][col_start+8] = 1
        
        # A1 * B4 -> C14
        U[a1][col_start+9] = 1
        V[b4][col_start+9] = 1
        W[c14][col_start+9] = 1

    # Global Indices
    # U: 2x5. U[i*5+j]
    # V: 5x5. V[j*5+k]
    # W: 5x2. W[k*2+i]
    
    # A_L: cols 0,1.
    # A_L[i,j] -> U[i*5+j]
    # A_L indices:
    # i=0: 0, 1
    # i=1: 5, 6
    # a00=0, a10=5, a01=1, a11=6
    
    # A_M: col 2.
    # A_M[i,0] -> U[i*5+2]
    # a0=2, a1=7
    
    # A_R: cols 3,4.
    # A_R[i,j] -> U[i*5+j+3]
    # i=0: 3, 4
    # i=1: 8, 9
    # a00=3, a10=8, a01=4, a11=9
    
    # B_T: rows 0,1.
    # B_T[j,k] -> V[j*5+k]
    # BT_1: j=0,1; k=0,1.
    # b00=0, b10=5, b01=1, b11=6
    # BT_2: j=0,1; k=2,3.
    # b00=2, b10=7, b01=3, b11=8
    # BT_3: j=0,1; k=4.
    # b0=4, b1=9
    
    # B_M: row 2.
    # B_M[0,k] -> V[2*5+k] = V[10+k]
    # b0=10, b1=11, b2=12, b3=13, b4=14
    
    # B_B: rows 3,4.
    # B_B[j,k] -> V[(j+3)*5+k]
    # BB_1: j=0,1 (local) -> 3,4 (global); k=0,1.
    # b00=15, b10=20, b01=16, b11=21
    # BB_2: j=0,1 -> 3,4; k=2,3.
    # b00=17, b10=22, b01=18, b11=23
    # BB_3: j=0,1 -> 3,4; k=4.
    # b0=19, b1=24
    
    # C: 2x5.
    # C[i,k] -> W[k*2+i]
    # C_L: k=0,1.
    # c00=0, c10=1, c01=2, c11=3
    # C_M: k=2.
    # c0=4, c1=5
    # C_R: k=3,4.
    # c00=6, c10=7, c01=8, c11=9
    
    # Wait, the split of C corresponds to the split of B's columns.
    # C = A_L B_T + A_M B_M + A_R B_B
    # B_T has cols 0,1,2,3,4? No.
    # B is 5x5.
    # B_T is top 2 rows of B. So B_T is 2x5.
    # B_M is middle 1 row of B. So B_M is 1x5.
    # B_B is bottom 2 rows of B. So B_B is 2x5.
    
    # A_L is left 2 cols of A. 2x2.
    # A_M is middle 1 col of A. 2x1.
    # A_R is right 2 cols of A. 2x2.
    
    # Product A_L B_T:
    # A_L is 2x2. B_T is 2x5.
    # Result is 2x5.
    # We split B_T into BT_1 (2x2), BT_2 (2x2), BT_3 (2x1).
    # BT_1 cols 0,1. BT_2 cols 2,3. BT_3 col 4.
    # So A_L BT_1 contributes to C cols 0,1.
    # A_L BT_2 contributes to C cols 2,3.
    # A_L BT_3 contributes to C col 4.
    
    # Similarly for A_R B_B.
    # A_R BB_1 contributes to C cols 0,1.
    # A_R BB_2 contributes to C cols 2,3.
    # A_R BB_3 contributes to C col 4.
    
    # And A_M B_M contributes to all C cols 0..4.
    
    # So we have contributions to each C column from multiple terms.
    # We just need to add them up in W.
    
    # Let's map the W indices for each sub-product.
    
    # 1. A_L * BT_1 (2x2 * 2x2) -> C cols 0,1
    # C[i,0] -> W[0*2+i] = W[i]
    # C[i,1] -> W[1*2+i] = W[2+i]
    # c_indices: [W0, W1, W2, W3] for [C00, C10, C01, C11]
    # c00=0, c10=1, c01=2, c11=3
    add_strassen([0, 5, 1, 6], [0, 5, 1, 6], [0, 1, 2, 3], 0)
    
    # 2. A_L * BT_2 (2x2 * 2x2) -> C cols 2,3
    # C[i,2] -> W[2*2+i] = W[4+i]
    # C[i,3] -> W[3*2+i] = W[6+i]
    # c_indices: [W4, W5, W6, W7]
    # B_T cols 2,3. V indices:
    # j=0: 2,3. j=1: 7,8.
    # b00=2, b10=7, b01=3, b11=8
    add_strassen([0, 5, 1, 6], [2, 7, 3, 8], [4, 5, 6, 7], 7)
    
    # 3. A_L * BT_3 (2x2 * 2x1) -> C col 4
    # C[i,4] -> W[4*2+i] = W[8+i]
    # c_indices: [W8, W9]
    # B_T col 4. V indices:
    # j=0: 4. j=1: 9.
    # b0=4, b1=9
    add_2x2x1([0, 5, 1, 6], [4, 9], [8, 9], 14)
    
    # 4. A_R * BB_1 (2x2 * 2x2) -> C cols 0,1
    # A_R indices: [3, 8, 4, 9]
    # B_B rows 3,4. Cols 0,1.
    # V indices:
    # j=3: 15,16. j=4: 20,21.
    # b00=15, b10=20, b01=16, b11=21
    # C cols 0,1 -> W[0..3]
    add_strassen([3, 8, 4, 9], [15, 20, 16, 21], [0, 1, 2, 3], 18)
    
    # 5. A_R * BB_2 (2x2 * 2x2) -> C cols 2,3
    # B_B rows 3,4. Cols 2,3.
    # V indices:
    # j=3: 17,18. j=4: 22,23.
    # b00=17, b10=22, b01=18, b11=23
    # C cols 2,3 -> W[4..7]
    add_strassen([3, 8, 4, 9], [17, 22, 18, 23], [4, 5, 6, 7], 25)
    
    # 6. A_R * BB_3 (2x2 * 2x1) -> C col 4
    # B_B rows 3,4. Col 4.
    # V indices:
    # j=3: 19. j=4: 24.
    # b0=19, b1=24
    # C col 4 -> W[8,9]
    add_2x2x1([3, 8, 4, 9], [19, 24], [8, 9], 32)
    
    # 7. A_M * B_M (2x1 * 1x5) -> C cols 0..4
    # A_M indices: [2, 7]
    # B_M row 2. Cols 0..4.
    # V indices: 10, 11, 12, 13, 14
    # C cols 0..4 -> W[0..9]
    add_2x1x5([2, 7], [10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 36)
    
    return {"U": U, "V": V, "W": W}

