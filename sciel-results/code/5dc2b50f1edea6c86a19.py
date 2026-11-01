import numpy as np

def construct():
    # 2x2 Strassen decomposition
    # U2 shape (4, 7), V2 shape (4, 7), W2 shape (4, 7)
    # Indices: U2[i*2+j, l], V2[j*2+k, l], W2[k*2+i, l]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a00+a11)(b00+b11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: M2 = (a10+a11)b00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[0][1] = 1; W2[3][1] = -1
    
    # Term 3: M3 = a00(b01-b11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 4: M4 = a11(b10-b00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 5: M5 = (a00+a01)b11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 6: M6 = (a10-a00)(b00+b01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7: M7 = (a01-a11)(b10+b11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 4x4 decomposition
    # Rank 49
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Coefficients for the 7 super-terms
    # alpha[l][i][j] for A block (i,j)
    # beta[l][j][k] for B block (j,k)
    # l is 0-indexed (0..6)
    
    alpha = [
        [[1,0],[0,1]], # P1: A00 + A11
        [[0,0],[1,1]], # P2: A10 + A11
        [[1,0],[0,0]], # P3: A00
        [[0,0],[0,1]], # P4: A11
        [[1,1],[0,0]], # P5: A00 + A01
        [[-1,0],[1,0]],# P6: A10 - A00
        [[0,1],[-1,0]] # P7: A01 - A11
    ]
    
    beta = [
        [[1,0],[0,1]], # P1: B00 + B11
        [[1,0],[0,0]], # P2: B00
        [[0,1],[-1,0]],# P3: B01 - B11
        [[-1,0],[1,0]],# P4: B10 - B00
        [[0,0],[0,1]], # P5: B11
        [[1,1],[0,0]], # P6: B00 + B01
        [[0,0],[1,1]]  # P7: B10 + B11
    ]
    
    for l in range(7):
        for m_idx in range(7):
            r = l * 7 + m_idx
            
            # Iterate over all entries in A, B, C
            # A index: i * 4 + j
            # B index: j * 4 + k
            # C index: k * 4 + i
            
            for i in range(4):
                for j in range(4):
                    # Map i, j to block coordinates
                    bi = i // 2
                    bj = j // 2
                    xi = i % 2
                    xj = j % 2
                    
                    coeff_a = alpha[l][bi][bj]
                    if coeff_a != 0:
                        # U2 index: xi * 2 + xj
                        u2_val = U2[xi * 2 + xj][m_idx]
                        if u2_val != 0:
                            U[i * 4 + j][r] = coeff_a * u2_val
                    
                    for k in range(4):
                        # Map j, k to block coordinates for B
                        # Note: j is shared.
                        # B block is (j_block, k_block)
                        # j is the column of A, row of B.
                        # So j_block = j // 2, xj = j % 2.
                        # k_block = k // 2, xk = k % 2.
                        
                        # V2 index: xj * 2 + xk
                        # Wait, V2 index is j' * 2 + k'.
                        # Here j' = xj, k' = xk.
                        
                        bk = k // 2
                        xk = k % 2
                        
                        coeff_b = beta[l][bj][bk]
                        if coeff_b != 0:
                            v2_val = V2[xj * 2 + xk][m_idx]
                            if v2_val != 0:
                                V[j * 4 + k][r] = coeff_b * v2_val
                        
                        # W2 index: xk * 2 + xi
                        # W index: k * 4 + i
                        w2_val = W2[xk * 2 + xi][m_idx]
                        if w2_val != 0:
                            W[k * 4 + i][r] = w2_val

    return {"U": U, "V": V, "W": W}
