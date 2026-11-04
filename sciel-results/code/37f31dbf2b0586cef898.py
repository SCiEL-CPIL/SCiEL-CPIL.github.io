import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen x Strassen)
    # Rank 8 for 4x4x2 (Schoolbook)
    # Total Rank 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Decomposition (Rank 49) ---
    # Strassen 2x2x2 decomposition
    # Indices: U[i,j], V[j,k], W[k,i] for i,j,k in 0..1
    # 7 terms
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M3 = A00(B10-B11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4 = A11(B01-B00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4x4 tensors via tensor product
    # U4[i1*2+i2, j1*2+j2, r1*7+r2] = U2[i1,j1,r1] * U2[i2,j2,r2]
    for r in range(49):
        r1 = r // 7
        r2 = r % 7
        for i in range(4):
            i1 = i // 2
            i2 = i % 2
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                if val != 0:
                    U[i*4+j][r] = val
                val = V2[j1*2+j1][r1] * V2[j2*2+j2][r2] # Wait, V2 indices are j,k
                # V2 row index is j*2+k? No, V2 is 4x7. Row j,k.
                # In 2x2, j in 0..1, k in 0..1.
                # V2[j*2+k]
                # In 4x4, J in 0..3, K in 0..3.
                # J = J1*2+J2, K = K1*2+K2.
                # V4[J, K, r] = V2[J1, K1, r1] * V2[J2, K2, r2]
                # Note: V2 is indexed by [j, k] flattened to j*2+k?
                # In my U2/V2/W2 definition above, I used flat indices 0..3.
                # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
                # So V2[j*2+k] is correct.
                
                # Correcting V4 construction:
                val_v = V2[j1*2+j1][r1] * V2[j2*2+j2][r2] # WRONG INDEXING IN LOGIC ABOVE
                # j1, k1 are the first level indices.
                # V2 row is j*2+k.
                # So V2[j1*2+k1]
                # My loop variables are i, j. For V, we need j, k.
                # But V is for B. B is 4x4 in this part.
                # So V4 rows are J, K (0..3).
                # We are iterating i, j for U. We need to iterate J, K for V.
                # Let's do it properly.

    # Redoing the filling of U, V, W for Part 1 cleanly
    for r in range(49):
        r1 = r // 7
        r2 = r % 7
        for i in range(4):
            i1 = i // 2
            i2 = i % 2
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                U[i*4+j][r] = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                
        for j in range(4):
            j1 = j // 2
            j2 = j % 2
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                V[j*4+k][r] = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                
        for k in range(4):
            k1 = k // 2
            k2 = k % 2
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                W[k*4+i][r] = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]

    # --- Part 2: 4x4x2 Decomposition (Rank 8) ---
    # Columns 49..56
    # U8[i, j, r] = delta(j, k) where r = c*4+k
    # V8[j, c, r] = delta(j, k)
    # W8[c, i, r] = 1
    for c in range(2):
        for k in range(4):
            r = 49 + c*4 + k
            for i in range(4):
                for j in range(4):
                    if j == k:
                        U[i*4+j][r] = 1
                    # V part: B is 4x2. Rows 16..23.
                    # V index: row = 16 + j, col = c
                    if j == k:
                        V[(16+j)*1 + c][r] = 1 # V is m*p = 4*6 = 24 rows.
                                                 # Row index for B: j (0..3), col c (0..1) -> j*2+c?
                                                 # No, B is 4x6. Row j, Col c.
                                                 # In V matrix (m*p x rank), row index is j*p + c?
                                                 # m=4, p=6. Row index = j*6 + c.
                                                 # So V[j*6+c][r]
                        V[j*6+c][r] = 1
                    # W part: C is 6x4. Row c (0..1), Col i (0..3)
                    # W index: c*4 + i
                    W[c*4+i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
