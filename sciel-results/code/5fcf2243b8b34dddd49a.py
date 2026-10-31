import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7x7 Strassen coefficients
    # beta[s][I][J], gamma[s][J][K], delta[s][I][K]
    # s is 0..6
    # I, J, K are 0..1
    
    # Initialize with zeros
    beta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    delta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # s=0 (M1)
    beta[0][0][0] = 1
    beta[0][1][1] = 1
    gamma[0][0][0] = 1
    gamma[0][1][1] = 1
    delta[0][0][0] = 1
    delta[0][1][1] = 1
    
    # s=1 (M2)
    beta[1][1][0] = 1
    beta[1][1][1] = 1
    gamma[1][0][0] = 1
    delta[1][1][0] = 1
    delta[1][1][1] = -1
    
    # s=2 (M3)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1
    gamma[2][1][1] = -1
    delta[2][0][1] = 1
    delta[2][1][1] = 1
    
    # s=3 (M4)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1
    gamma[3][0][0] = -1
    delta[3][0][0] = 1
    delta[3][1][0] = 1
    
    # s=4 (M5)
    beta[4][0][0] = 1
    beta[4][0][1] = 1
    gamma[4][1][1] = 1
    delta[4][0][0] = -1
    delta[4][0][1] = 1
    
    # s=5 (M6)
    beta[5][1][0] = 1
    beta[5][0][0] = -1
    gamma[5][0][0] = 1
    gamma[5][0][1] = 1
    delta[5][1][1] = 1
    
    # s=6 (M7)
    beta[6][0][1] = 1
    beta[6][1][1] = -1
    gamma[6][1][0] = 1
    gamma[6][1][1] = 1
    delta[6][0][0] = 1
    
    # Define U2, V2, W2 for 2x2 Strassen (rank 7)
    # Rows: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # Cols: 0..6
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # t=0 (M1 inner)
    # X1 = A00 + A11, Y1 = B00 + B11
    # C00 += M1, C11 += M1
    # U2: A00(0), A11(3) are 1
    U2[0][0] = 1
    U2[3][0] = 1
    # V2: B00(0), B11(3) are 1
    V2[0][0] = 1
    V2[3][0] = 1
    # W2: C00(0), C11(3) are 1
    W2[0][0] = 1
    W2[3][0] = 1
    
    # t=1 (M2 inner)
    # X2 = A10 + A11, Y2 = B00
    # C10 += M2, C11 -= M2
    U2[2][1] = 1 # A10
    U2[3][1] = 1 # A11
    V2[0][1] = 1 # B00
    W2[1][1] = 1 # C10 (idx 1)
    W2[3][1] = -1 # C11 (idx 3)
    
    # t=2 (M3 inner)
    # X3 = A00, Y3 = B01 - B11
    # C01 += M3, C11 += M3
    U2[0][2] = 1 # A00
    V2[1][2] = 1 # B01
    V2[3][2] = -1 # B11
    W2[2][2] = 1 # C01 (idx 2)
    W2[3][2] = 1 # C11 (idx 3)
    
    # t=3 (M4 inner)
    # X4 = A11, Y4 = B10 - B00
    # C00 += M4, C10 += M4
    U2[3][3] = 1 # A11
    V2[2][3] = 1 # B10
    V2[0][3] = -1 # B00
    W2[0][3] = 1 # C00 (idx 0)
    W2[1][3] = 1 # C10 (idx 1)
    
    # t=4 (M5 inner)
    # X5 = A00 + A01, Y5 = B11
    # C00 -= M5, C01 += M5
    U2[0][4] = 1 # A00
    U2[1][4] = 1 # A01
    V2[3][4] = 1 # B11
    W2[0][4] = -1 # C00 (idx 0)
    W2[2][4] = 1 # C01 (idx 2)
    
    # t=5 (M6 inner)
    # X6 = A10 - A00, Y6 = B00 + B01
    # C11 += M6
    U2[2][5] = 1 # A10
    U2[0][5] = -1 # A00
    V2[0][5] = 1 # B00
    V2[1][5] = 1 # B01
    W2[3][5] = 1 # C11 (idx 3)
    
    # t=6 (M7 inner)
    # X7 = A01 - A11, Y7 = B10 + B11
    # C00 += M7
    U2[1][6] = 1 # A01
    U2[3][6] = -1 # A11
    V2[2][6] = 1 # B10
    V2[3][6] = 1 # B11
    W2[0][6] = 1 # C00 (idx 0)
    
    # Construct U, V, W
    # Loop over s (0..6) and t (0..6)
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            # Loop over A indices
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    # Coefficient is beta[s][I][J] * U2[i_prime*2 + j_prime][t]
                    coeff = beta[s][I][J] * U2[i_prime * 2 + j_prime][t]
                    if coeff != 0:
                        U[row_u][c] = coeff
                        
            # Loop over B indices
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    # Coefficient is gamma[s][J][K] * V2[j_prime*2 + k_prime][t]
                    coeff = gamma[s][J][K] * V2[j_prime * 2 + k_prime][t]
                    if coeff != 0:
                        V[row_v][c] = coeff
                        
            # Loop over C indices
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    # Coefficient is delta[s][I][K] * W2[k_prime*2 + i_prime][t]
                    coeff = delta[s][I][K] * W2[k_prime * 2 + i_prime][t]
                    if coeff != 0:
                        W[row_w][c] = coeff
                        
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
