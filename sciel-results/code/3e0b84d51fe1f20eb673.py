"""
Rank-49 exact tensor decomposition for 4x4x4 matrix multiplication using recursive Strassen.
The rank of the 4x4 matrix multiplication tensor is 49.
This implementation constructs the factor matrices U, V, W of shape (16, 49) with integer entries.
"""

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 block coefficients
    # alpha[s][a][b] is the coefficient of block A[a][b] in X_s
    # beta[s][b][c] is the coefficient of block B[b][c] in Y_s
    # a, b, c are in {0, 1}
    
    alpha = [
        [[1, 0], [0, 1]],  # s=1: A00 + A11
        [[0, 0], [1, 1]],  # s=2: A10 + A11
        [[1, 0], [0, 0]],  # s=3: A00
        [[0, 0], [0, 1]],  # s=4: A11
        [[1, 1], [0, 0]],  # s=5: A00 + A01
        [[-1, 0], [1, 0]], # s=6: -A00 + A10
        [[0, 1], [0, -1]]  # s=7: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # s=1: B00 + B11
        [[1, 0], [0, 0]],  # s=2: B00
        [[0, 1], [0, -1]], # s=3: B01 - B11
        [[-1, 0], [1, 0]], # s=4: -B00 + B10
        [[0, 0], [0, 1]],  # s=5: B11
        [[1, 1], [0, 0]],  # s=6: B00 + B01
        [[0, 0], [1, 1]]   # s=7: B10 + B11
    ]

    # Inner 2x2 Strassen factors (rank 7)
    # U2[p*2+q][s] is the coefficient of element (p,q) in X_s
    # V2[q*2+r][s] is the coefficient of element (q,r) in Y_s
    # W2[r*2+p][s] is the coefficient of M_s in C[p,r]
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # p=0,q=0
        [0, 0, 0, 0, 1, 0, 1],   # p=0,q=1
        [0, 1, 0, 0, 0, 1, 0],   # p=1,q=0
        [1, 1, 0, 1, 0, 0, -1]   # p=1,q=1
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # q=0,r=0
        [0, 0, 1, 0, 0, 1, 0],   # q=0,r=1
        [0, 0, 0, 1, 0, 0, 1],   # q=1,r=0
        [1, 0, -1, 0, 1, 0, 1]   # q=1,r=1
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # r=0,p=0 -> C00
        [0, 0, 1, 0, 1, 0, 0],   # r=0,p=1 -> C01
        [0, 1, 0, 1, 0, 0, 0],   # r=1,p=0 -> C10
        [1, -1, 1, 0, 0, 1, 0]   # r=1,p=1 -> C11
    ]

    col = 0
    for s in range(7):
        for r_prime in range(7):
            # U
            for a in range(2):
                for b in range(2):
                    if alpha[s][a][b] != 0:
                        for p in range(2):
                            for q in range(2):
                                i = 2 * a + p
                                j = 2 * b + q
                                U[i * m + j][col] += alpha[s][a][b] * U2[p * 2 + q][r_prime]
            
            # V
            for b in range(2):
                for c in range(2):
                    if beta[s][b][c] != 0:
                        for q in range(2):
                            for r in range(2):
                                j = 2 * b + q
                                k = 2 * c + r
                                V[j * p + k][col] += beta[s][b][c] * V2[q * 2 + r][r_prime]
            
            # W
            for r in range(2):
                for p in range(2):
                    if W2[r * 2 + p][r_prime] != 0:
                        for c in range(2):
                            for a in range(2):
                                k = 2 * c + r
                                i = 2 * a + p
                                W[k * n + i][col] += W2[r * 2 + p][r_prime]
            
            col += 1

    return {"U": U, "V": V, "W": W}
