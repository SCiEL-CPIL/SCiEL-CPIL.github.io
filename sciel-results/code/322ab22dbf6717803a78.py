import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # 6x6 matrices as 3x3 blocks of 2x2 matrices.
    # Use Schoolbook for 3x3 block multiplication (Rank 27).
    # Use Strassen for 2x2 sub-block multiplication (Rank 7).
    # Total Rank = 27 * 7 = 189.
    
    rank_3x3 = 27
    rank_2x2 = 7
    total_rank = rank_3x3 * rank_2x2
    
    # Initialize U, V, W
    # U: (n*m, total_rank) = (36, 189)
    # V: (m*p, total_rank) = (36, 189)
    # W: (p*n, total_rank) = (36, 189)
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # 1. Define 3x3 Schoolbook Decomposition (Rank 27)
    # U3: 9x27, V3: 9x27, W3: 9x27
    U3 = [[0] * rank_3x3 for _ in range(9)]
    V3 = [[0] * rank_3x3 for _ in range(9)]
    W3 = [[0] * rank_3x3 for _ in range(9)]
    
    for r in range(rank_3x3):
        i = r // 9
        j = (r // 3) % 3
        k = r % 3
        U3[i * 3 + j][r] = 1
        V3[j * 3 + k][r] = 1
        W3[k * 3 + i][r] = 1
        
    # 2. Define 2x2 Strassen Decomposition (Rank 7)
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [[0] * rank_2x2 for _ in range(4)]
    V2 = [[0] * rank_2x2 for _ in range(4)]
    W2 = [[0] * rank_2x2 for _ in range(4)]
    
    # Strassen M1..M7
    # M1: (a+d)(e+h)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00 += M1, C11 += M1
    
    # M2: (c+d)e
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1 # C10 += M2, C11 -= M2
    
    # M3: a(f-e)
    U2[0][2] = 1
    V2[1][2] = 1; V2[0][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01 += M3, C11 += M3
    
    # M4: d(g-e)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1 # C00 += M4
    
    # M5: (a+b)(h-g)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1; V2[2][4] = -1
    W2[0][4] = -1; W2[2][4] = 1 # C00 -= M5, C10 += M5
    
    # M6: (c+a)(e+f)
    U2[2][5] = 1; U2[0][5] = 1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11 += M6
    
    # M7: (b+d)(g+h)
    U2[1][6] = 1; U2[3][6] = 1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1; W2[1][6] = 1 # C00 -= M7, C01 += M7
    
    # 3. Construct Full 6x6 Decomposition
    for r in range(total_rank):
        r3 = r // rank_2x2
        r2 = r % rank_2x2
        
        # Fill U
        for I in range(3):
            for J in range(3):
                coeff3 = U3[I * 3 + J][r3]
                if coeff3 != 0:
                    for ip in range(2):
                        for jp in range(2):
                            coeff2 = U2[ip * 2 + jp][r2]
                            if coeff2 != 0:
                                i = 2 * I + ip
                                j = 2 * J + jp
                                U[i * 6 + j][r] = coeff3 * coeff2
                                
        # Fill V
        for J in range(3):
            for K in range(3):
                coeff3 = V3[J * 3 + K][r3]
                if coeff3 != 0:
                    for jp in range(2):
                        for kp in range(2):
                            coeff2 = V2[jp * 2 + kp][r2]
                            if coeff2 != 0:
                                j = 2 * J + jp
                                k = 2 * K + kp
                                V[j * 6 + k][r] = coeff3 * coeff2
                                
        # Fill W
        for K in range(3):
            for I in range(3):
                coeff3 = W3[K * 3 + I][r3]
                if coeff3 != 0:
                    for kp in range(2):
                        for ip in range(2):
                            coeff2 = W2[kp * 2 + ip][r2]
                            if coeff2 != 0:
                                k = 2 * K + kp
                                i = 2 * I + ip
                                W[k * 6 + i][r] = coeff3 * coeff2

    return {"U": U, "V": V, "W": W}
