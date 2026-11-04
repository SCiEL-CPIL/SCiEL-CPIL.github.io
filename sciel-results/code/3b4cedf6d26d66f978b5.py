import numpy as np

def construct():
    n, m, p = 3, 5, 6
    total_rank = 27
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 2x5x6 Block (Rank 21) ---
    # Based on Kronecker product of Strassen (2x2x2, rank 7) and 1x5x3 (rank 3)
    
    # Strassen Factors (2x2x2)
    # U1 (2x7), V1 (2x7), W1 (2x7)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A10)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    U1 = [[0]*7 for _ in range(2)]
    V1 = [[0]*7 for _ in range(2)]
    W1 = [[0]*7 for _ in range(2)]
    
    # M1
    U1[0][0] = 1; U1[1][0] = 1
    V1[0][0] = 1; V1[1][0] = 1
    W1[0][0] = 1; W1[1][0] = 1
    
    # M2
    U1[1][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1
    
    # M3
    U1[0][2] = 1
    V1[1][2] = 1; V1[0][2] = -1
    W1[0][2] = 1
    
    # M4
    U1[1][3] = 1
    V1[1][3] = 1; V1[0][3] = -1
    W1[1][3] = 1
    
    # M5
    U1[0][4] = 1; U1[1][4] = 1
    V1[1][4] = 1
    W1[1][4] = 1
    
    # M6
    U1[1][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[1][5] = 1
    
    # M7
    U1[0][6] = 1; U1[1][6] = -1
    V1[1][6] = 1; V1[0][6] = 1
    W1[1][6] = 1
    
    # 1x5x3 Factors (Rank 3)
    # U2 (5x3), V2 (15x3), W2 (3x3)
    U2 = [[0]*3 for _ in range(5)]
    V2 = [[0]*3 for _ in range(15)]
    W2 = [[0]*3 for _ in range(3)]
    
    for k in range(3):
        for j in range(5):
            U2[j][k] = 1
            V2[j*3 + k][k] = 1
        W2[k][k] = 1
        
    # Construct 2x5x6 Factors from Kronecker
    # U_total (10x21), V_total (30x21), W_total (12x21)
    # But we only need to fill the global U, V, W for the 2x5x6 part
    
    # Mapping:
    # U index: i (0,1) * 5 + j (0..4)
    # V index: j (0..4) * 6 + k (0..5)
    # W index: k (0..5) * 2 + i (0,1)
    
    # Kronecker indices:
    # i = i1 * 1 + i2 (i2=0) -> i1
    # j = j1 * 5 + j2
    # k = k1 * 3 + k2
    
    # Wait, the Kronecker product logic for the factors:
    # U_kronecker[i1, i2, r1, r2] = U1[i1, r1] * U2[i2, r2]
    # Since i2 is always 0, U_kronecker[i1, 0, r1, r2] = U1[i1, r1] * U2[0, r2]
    # But U2 is 5x3. The index i2 corresponds to the row of U2.
    # In the 1x5x3 tensor, the first dimension is 1. So i2=0.
    # So U2[0, r2] is the only row.
    # U2[0, r2] = 1 for all r2 (since U2[j][k]=1).
    # So U_kronecker[i1, 0, r1, r2] = U1[i1, r1] * 1 = U1[i1, r1].
    
    # However, the row index in the global U for the 2x5x6 block is i*5 + j.
    # i is the row of A (0,1).
    # j is the column of A (0..4).
    # In the Kronecker product, j = j1*5 + j2.
    # j1 comes from 2x2x2 (0,1).
    # j2 comes from 1x5x3 (0..4).
    
    # So for a given global row index `row_u` in 0..9:
    # i = row_u // 5
    # j = row_u % 5
    # This j corresponds to the "middle" dimension of the 2x5x6 product.
    # But in the Kronecker construction, the middle dimension is 10 (2*5).
    # We are truncating to 5.
    # The rows of V_total correspond to j in 0..9.
    # We take j in 0..4.
    # These j correspond to j1=0, j2=0..4.
    # So we only use the part of the Kronecker product where j1=0.
    
    # This implies that we only use the terms in the Strassen decomposition where the index j1=0 is active?
    # No, the Strassen decomposition is for 2x2x2.
    # The Kronecker product combines them.
    # If we restrict j to 0..4, we are restricting the "j" dimension of the 2x10x6 tensor.
    # The j dimension of 2x10x6 is formed by j1 (from 2x2) and j2 (from 1x5).
    # j = j1 * 5 + j2.
    # If j < 5, then j1 must be 0.
    # So we only use the part of the tensor where j1=0.
    
    # Does the tensor T_{2,10,6} have non-zero entries only when j1=0?
    # No, T_{2,10,6} is the full multiplication tensor.
    # But we are embedding T_{2,5,6} into it.
    # The embedding sets the inputs for j1=1 to 0.
    # This means the contribution from the j1=1 part is 0.
    # So we can ignore the terms that involve j1=1 in the V matrix?
    # V_total has rows for all j.
    # We just take the rows for j=0..4.
    # These rows correspond to j1=0.
    # So we only need the columns of V_total that correspond to j1=0?
    # No, V_total is a matrix. The rows are the indices.
    # We just slice the rows.
    
    # So, for the 2x5x6 block:
    # We use the Kronecker product factors, but we slice V to the first 30 rows (j=0..4, k=0..5).
    # And we slice U?
    # U_total rows are i*10 + j.
    # We want i*5 + j.
    # The rows of U_total for j=0..4 and j=5..9 are distinct.
    # But U_total depends on i and j.
    # U_total[i*10 + j, r] = U1[i1, r1] * U2[j2, r2]?
    # No, U index is (i1, i2).
    # i1 is from 2x2. i2 is from 1x5 (always 0).
    # So U_total has 2*1=2 distinct i patterns?
    # No, U is n*m.
    # n=2, m=10.
    # Rows are 0..19.
    # Row index = i*10 + j.
    # i in 0,1. j in 0..9.
    # U_total[i*10 + j, r] = U1[i, r1] * U2[j2, r2] ??
    # The Kronecker product of U1 (2x7) and U2 (5x3) is a 10x21 matrix.
    # Row index is i1*5 + i2.
    # Here i1 is the row of U1 (0,1). i2 is the row of U2 (0..4).
    # So the row index in U_total is i1*5 + i2.
    # This matches the row index in U for 2x5x6 (i*5 + j).
    # So we can just use the full U_total (10x21).
    
    # Similarly for W.
    # W_total is 12x21.
    # W index is k1*3 + k2.
    # k1 is col of W1 (0,1). k2 is col of W2 (0..2).
    # Wait, W is p*n.
    # p=6, n=2.
    # W index is k*2 + i.
    # In Kronecker: k = k1*3 + k2.
    # i = i1*1 + i2.
    # So W_total row index is k*2 + i = (k1*3 + k2)*2 + (i1*1 + 0) = k1*6 + k2*2 + i1.
    # This matches the row index in W for 2x5x6 (k*2 + i).
    # So we use the full W_total (12x21).
    
    # So we just need to compute the Kronecker product of the factors.
    
    for i1 in range(2):
        for i2 in range(5): # i2 is j
            row_u = i1 * 5 + i2
            for r1 in range(7):
                for r2 in range(3):
                    r = r1 * 3 + r2
                    U[row_u][r] = U1[i1][r1] * U2[i2][r2]
                    
    for j1 in range(2): # j1 is 0 or 1, but we only use j1=0 for the 5x6 part?
        # Wait, V index is j*6 + k.
        # j = j1*5 + j2.
        # k = k1*3 + k2.
        # We only want j < 5, so j1=0.
        for j2 in range(5):
            j = 0 * 5 + j2
            for k1 in range(2):
                for k2 in range(3):
                    k = k1 * 3 + k2
                    row_v = j * 6 + k
                    for r1 in range(7):
                        for r2 in range(3):
                            r = r1 * 3 + r2
                            V[row_v][r] = V1[j1][r1] * V2[j2*3 + k2][r2]
                            # Note: V2 index is j2*3 + k2?
                            # V2 is 15x3. Row index is j2*3 + k2.
                            # Yes.
                            
    for k1 in range(2):
        for k2 in range(3):
            k = k1 * 3 + k2
            for i1 in range(2):
                row_w = k * 2 + i1
                for r1 in range(7):
                    for r2 in range(3):
                        r = r1 * 3 + r2
                        W[row_w][r] = W1[i1][r1] * W2[k2][r2]
                        # W2 index is k2.
                        # W1 index is i1.
                        # W_total row index is k*2 + i1.
                        # Yes.

    # --- Part 2: 1x5x6 Block (Rank 6) ---
    # Offset rank by 21
    offset = 21
    
    # U2_part (5x6), V2_part (30x6), W2_part (6x6)
    # Rows of U: 10..14
    # Rows of V: 0..29
    # Rows of W: 12..17
    
    for k in range(6):
        for j in range(5):
            U[10 + j][offset + k] = 1
            V[j * 6 + k][offset + k] = 1
        W[12 + k][offset + k] = 1
        
    return {"U": U, "V": V, "W": W}
