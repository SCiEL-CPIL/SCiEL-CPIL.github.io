import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 43 decomposition
    # 7 (Strassen 2x2) + 4 (a_col b_row^T) for C11 of B1
    # 4 (A22 b_col) + 2 (a_col b_bot) for C12 of B1
    # 4 (a_row^T B22) + 2 (a_bot b_row^T) for C21 of B1
    # 2 (a_row^T b_col) + 1 (a_bot b_bot) for C22 of B1
    # 7 (Strassen 2x2) + 4 (a_col y^T) for C1,4:5
    # 4 (a_row^T X) + 2 (a_bot y^T) for C2,4:5
    # Total: 11 + 6 + 6 + 3 + 11 + 6 = 43
    
    rank = 43
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1
        
    # Indices
    # A: i*m + j
    # B: j*p + k
    # C: k*n + i
    
    # Part 1: B1 (k=0,1,2)
    # C11 (i=0,1; k=0,1)
    # Strassen for A22 * B22
    # A22 indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B22 indices: (0,0)->0, (0,1)->5, (1,0)->1, (1,1)->6
    
    # M1
    add_term(0, 0, 0, 1, 1, 1) # A00 B00
    add_term(4, 6, 0, 1, 1, 1) # A11 B11
    add_term(0, 6, 0, 1, 1, -1) # A00 B11 (subtracted in C00? No, M1 is sum)
    # Wait, Strassen factors are linear combinations.
    # We need to add columns for each product in the linear combination.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This expands to 4 rank-1 terms.
    # But Strassen uses 7 products, not 16.
    # The products are:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01+B11)
    # P5 = (A00+A01)(B11-B10)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We need to represent these as columns in U, V, W.
    # Each Pk is a product of a linear form in A and a linear form in B.
    # So for each Pk, we add one column where U has the coeffs of the A-form, V has coeffs of the B-form, and W has the contribution to C.
    
    # Let's redefine add_term to take vectors.
    def add_strassen_term(u_vec, v_vec, w_vec):
        nonlocal col
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
        col += 1

    # Reset col to 0
    col = 0
    
    # Strassen 2x2 for A22 B22
    # A indices: 0,1,3,4
    # B indices: 0,5,1,6
    
    # P1
    u1 = np.zeros(9); u1[0] = 1; u1[4] = 1
    v1 = np.zeros(15); v1[0] = 1; v1[6] = 1
    w1 = np.zeros(15); w1[0] = 1; w1[3] = 1; w1[6] = 1; w1[9] = 1 # C00, C01, C10, C11 contributions?
    # C00 += P1, C01 += 0, C10 += 0, C11 += P1?
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So W for P1 should have 1 at C00 and C11.
    # C00 index: k=0, i=0 -> 0*3+0=0
    # C11 index: k=1, i=1 -> 1*3+1=4? No, C is 5x3.
    # C indices: k*n + i.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 3
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 4
    
    w1 = np.zeros(15)
    w1[0] = 1 # C00
    w1[4] = 1 # C11
    add_strassen_term(u1, v1, w1)
    
    # P2 = (A10+A11)B00
    u2 = np.zeros(9); u2[3] = 1; u2[4] = 1
    v2 = np.zeros(15); v2[0] = 1
    w2 = np.zeros(15)
    w2[1] = 1 # C10
    w2[4] = -1 # C11
    add_strassen_term(u2, v2, w2)
    
    # P3 = A00(B10-B11)
    u3 = np.zeros(9); u3[0] = 1
    v3 = np.zeros(15); v3[1] = 1; v3[6] = -1
    w3 = np.zeros(15)
    w3[3] = 1 # C01
    w3[4] = 1 # C11
    add_strassen_term(u3, v3, w3)
    
    # P4 = A11(B01+B11)
    u4 = np.zeros(9); u4[4] = 1
    v4 = np.zeros(15); v4[5] = 1; v4[6] = 1
    w4 = np.zeros(15)
    w4[0] = 1 # C00
    w4[1] = 1 # C10
    add_strassen_term(u4, v4, w4)
    
    # P5 = (A00+A01)(B11-B10)
    u5 = np.zeros(9); u5[0] = 1; u5[1] = 1
    v5 = np.zeros(15); v5[6] = 1; v5[1] = -1
    w5 = np.zeros(15)
    w5[0] = -1 # C00
    w5[3] = 1 # C01
    add_strassen_term(u5, v5, w5)
    
    # P6 = (A10-A00)(B00+B01)
    u6 = np.zeros(9); u6[3] = 1; u6[0] = -1
    v6 = np.zeros(15); v6[0] = 1; v6[5] = 1
    w6 = np.zeros(15)
    w6[4] = 1 # C11
    add_strassen_term(u6, v6, w6)
    
    # P7 = (A01-A11)(B10+B11)
    u7 = np.zeros(9); u7[1] = 1; u7[4] = -1
    v7 = np.zeros(15); v7[1] = 1; v7[6] = 1
    w7 = np.zeros(15)
    w7[0] = 1 # C00
    add_strassen_term(u7, v7, w7)
    
    # Now trivial terms for C11 part: a_col b_row^T
    # a_col: A(0,2)=2, A(1,2)=5
    # b_row: B(2,0)=10, B(2,1)=11
    # Products: A02 B20, A02 B21, A12 B20, A12 B21
    # C00 += A02 B20
    # C01 += A02 B21
    # C10 += A12 B20
    # C11 += A12 B21
    
    add_term(2, 10, 0, 1, 1, 1) # A02 B20 -> C00
    add_term(2, 11, 3, 1, 1, 1) # A02 B21 -> C01
    add_term(5, 10, 1, 1, 1, 1) # A12 B20 -> C10
    add_term(5, 11, 4, 1, 1, 1) # A12 B21 -> C11
    
    # C12 part (i=0,1; k=2)
    # A22 b_col (rank 4)
    # A00 B02, A01 B12, A10 B02, A11 B12
    # C02 += A00 B02 + A01 B12
    # C12 += A10 B02 + A11 B12
    add_term(0, 2, 6, 1, 1, 1) # A00 B02 -> C02 (k=2,i=0 -> 2*3+0=6)
    add_term(1, 7, 6, 1, 1, 1) # A01 B12 -> C02 (B12 index: j=1,k=2 -> 1*5+2=7)
    add_term(3, 2, 7, 1, 1, 1) # A10 B02 -> C12 (k=2,i=1 -> 2*3+1=7)
    add_term(4, 7, 7, 1, 1, 1) # A11 B12 -> C12
    
    # a_col b_bot (rank 2)
    # A02 B22, A12 B22
    # C02 += A02 B22
    # C12 += A12 B22
    add_term(2, 12, 6, 1, 1, 1) # A02 B22 (B22 index: j=2,k=2 -> 2*5+2=12)
    add_term(5, 12, 7, 1, 1, 1) # A12 B22
    
    # C21 part (i=2; k=0,1)
    # a_row^T B22 (rank 4)
    # A20 B00, A20 B01, A21 B10, A21 B11
    # C20 += A20 B00 + A21 B10
    # C21 += A20 B01 + A21 B11
    add_term(6, 0, 8, 1, 1, 1) # A20 B00 -> C20 (k=0,i=2 -> 0*3+2=2? No, k*n+i. 0*3+2=2)
    # Wait, C indices: k*n + i.
    # C20: k=0, i=2 -> 2
    # C21: k=1, i=2 -> 5
    # C02: k=2, i=0 -> 6
    # C12: k=2, i=1 -> 7
    # C22: k=2, i=2 -> 8
    
    # Correcting W indices in previous steps:
    # C00: 0
    # C01: 3
    # C10: 1
    # C11: 4
    # C02: 6
    # C12: 7
    # C20: 2
    # C21: 5
    # C22: 8
    
    # Re-evaluating W indices for Strassen:
    # P1: C00(0), C11(4)
    # P2: C10(1), C11(4)
    # P3: C01(3), C11(4)
    # P4: C00(0), C10(1)
    # P5: C00(0), C01(3)
    # P6: C11(4)
    # P7: C00(0)
    
    # My previous W assignments were mostly correct, but let's double check C10/C20.
    # C10 is k=0, i=1 -> 1.
    # C20 is k=0, i=2 -> 2.
    
    # The code above for C21 part:
    add_term(6, 0, 2, 1, 1, 1) # A20 B00 -> C20
    add_term(6, 5, 5, 1, 1, 1) # A20 B01 -> C21 (B01 index: j=0,k=1 -> 1)
    add_term(7, 1, 2, 1, 1, 1) # A21 B10 -> C20 (B10 index: j=1,k=0 -> 5)
    add_term(7, 6, 5, 1, 1, 1) # A21 B11 -> C21 (B11 index: j=1,k=1 -> 6)
    
    # a_bot b_row^T (rank 2)
    # A22 B20, A22 B21
    # C20 += A22 B20
    # C21 += A22 B21
    add_term(8, 10, 2, 1, 1, 1) # A22 B20
    add_term(8, 11, 5, 1, 1, 1) # A22 B21
    
    # C22 part (i=2; k=2)
    # a_row^T b_col (rank 2)
    # A20 B02, A21 B12
    # C22 += A20 B02 + A21 B12
    add_term(6, 2, 8, 1, 1, 1) # A20 B02
    add_term(7, 7, 8, 1, 1, 1) # A21 B12
    
    # a_bot b_bot (rank 1)
    # A22 B22
    # C22 += A22 B22
    add_term(8, 12, 8, 1, 1, 1) # A22 B22
    
    # Part 2: B2 (k=3,4)
    # C1,4:5 (i=0,1; k=3,4)
    # A22 X (Strassen 2x2)
    # X indices: B(0,3)=3, B(0,4)=4, B(1,3)=8, B(1,4)=9
    # C03 (k=3,i=0 -> 9), C04 (k=4,i=0 -> 12)
    # C13 (k=3,i=1 -> 10), C14 (k=4,i=1 -> 13)
    
    # P1'
    u1p = np.zeros(9); u1p[0] = 1; u1p[4] = 1
    v1p = np.zeros(15); v1p[3] = 1; v1p[9] = 1
    w1p = np.zeros(15); w1p[9] = 1; w1p[13] = 1 # C03, C14
    add_strassen_term(u1p, v1p, w1p)
    
    # P2'
    u2p = np.zeros(9); u2p[3] = 1; u2p[4] = 1
    v2p = np.zeros(15); v2p[3] = 1
    w2p = np.zeros(15); w2p[10] = 1; w2p[13] = -1 # C13, C14
    add_strassen_term(u2p, v2p, w2p)
    
    # P3'
    u3p = np.zeros(9); u3p[0] = 1
    v3p = np.zeros(15); v3p[8] = 1; v3p[9] = -1
    w3p = np.zeros(15); w3p[12] = 1; w3p[13] = 1 # C04, C14
    add_strassen_term(u3p, v3p, w3p)
    
    # P4'
    u4p = np.zeros(9); u4p[4] = 1
    v4p = np.zeros(15); v4p[4] = 1; v4p[9] = 1
    w4p = np.zeros(15); w4p[9] = 1; w4p[10] = 1 # C03, C13
    add_strassen_term(u4p, v4p, w4p)
    
    # P5'
    u5p = np.zeros(9); u5p[0] = 1; u5p[1] = 1
    v5p = np.zeros(15); v5p[9] = 1; v5p[8] = -1
    w5p = np.zeros(15); w5p[9] = -1; w5p[12] = 1 # C03, C04
    add_strassen_term(u5p, v5p, w5p)
    
    # P6'
    u6p = np.zeros(9); u6p[3] = 1; u6p[0] = -1
    v6p = np.zeros(15); v6p[3] = 1; v6p[4] = 1
    w6p = np.zeros(15); w6p[13] = 1 # C14
    add_strassen_term(u6p, v6p, w6p)
    
    # P7'
    u7p = np.zeros(9); u7p[1] = 1; u7p[4] = -1
    v7p = np.zeros(15); v7p[8] = 1; v7p[9] = 1
    w7p = np.zeros(15); w7p[9] = 1 # C03
    add_strassen_term(u7p, v7p, w7p)
    
    # a_col y^T (rank 4)
    # A02 B23, A02 B24, A12 B23, A12 B24
    # B23 index: j=2,k=3 -> 13
    # B24 index: j=2,k=4 -> 14
    # C03 += A02 B23
    # C04 += A02 B24
    # C13 += A12 B23
    # C14 += A12 B24
    add_term(2, 13, 9, 1, 1, 1)
    add_term(2, 14, 12, 1, 1, 1)
    add_term(5, 13, 10, 1, 1, 1)
    add_term(5, 14, 13, 1, 1, 1)
    
    # C2,4:5 (i=2; k=3,4)
    # a_row^T X (rank 4)
    # A20 B03, A20 B04, A21 B13, A21 B14
    # C23 (k=3,i=2 -> 11)
    # C24 (k=4,i=2 -> 14)
    add_term(6, 3, 11, 1, 1, 1)
    add_term(6, 4, 14, 1, 1, 1)
    add_term(7, 8, 11, 1, 1, 1)
    add_term(7, 9, 14, 1, 1, 1)
    
    # a_bot y^T (rank 2)
    # A22 B23, A22 B24
    add_term(8, 13, 11, 1, 1, 1)
    add_term(8, 14, 14, 1, 1, 1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
