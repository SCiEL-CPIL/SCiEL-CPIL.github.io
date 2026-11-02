import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base 2x2 Strassen factors
    # U2[i*2+j, s], V2[j*2+k, s], W2[k*2+i, s]
    # s in 0..6
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # m1 = A00 * (B00 + B11)
    # A00 is i=0,j=0 -> U2[0,0]=1
    # B00 is j=0,k=0 -> V2[0,0]=1
    # B11 is j=1,k=1 -> V2[3,0]=1
    # Contributes to C00 (k=0,i=0 -> W2[0,0]) and C11 (k=1,i=1 -> W2[3,0])
    U2[0][0] = 1
    V2[0][0] = 1
    V2[3][0] = 1
    W2[0][0] = 1
    W2[3][0] = 1
    
    # m2 = (A00 + A10) * B11
    # A00 -> U2[0,1]=1
    # A10 is i=1,j=0 -> U2[2,1]=1
    # B11 is j=1,k=1 -> V2[3,1]=1
    # Contributes to C10 (k=1,i=0 -> W2[2,1]) and C11 (k=1,i=1 -> W2[3,1]) with -1
    U2[0][1] = 1
    U2[2][1] = 1
    V2[3][1] = 1
    W2[2][1] = 1
    W2[3][1] = -1
    
    # m3 = (A01 - A11) * (B00 + B01)
    # A01 is i=0,j=1 -> U2[1,2]=1
    # A11 is i=1,j=1 -> U2[3,2]=-1
    # B00 is j=0,k=0 -> V2[0,2]=1
    # B01 is j=0,k=1 -> V2[1,2]=1
    # Contributes to C01 (k=1,i=0 -> W2[2,2]) and C11 (k=1,i=1 -> W2[3,2])
    U2[1][2] = 1
    U2[3][2] = -1
    V2[0][2] = 1
    V2[1][2] = 1
    W2[2][2] = 1
    W2[3][2] = 1
    
    # m4 = A11 * (B10 - B00)
    # A11 is i=1,j=1 -> U2[3,3]=1
    # B10 is j=1,k=0 -> V2[2,3]=1
    # B00 is j=0,k=0 -> V2[0,3]=-1
    # Contributes to C00 (k=0,i=0 -> W2[0,3]) and C10 (k=1,i=0 -> W2[2,3])
    U2[3][3] = 1
    V2[2][3] = 1
    V2[0][3] = -1
    W2[0][3] = 1
    W2[2][3] = 1
    
    # m5 = (A00 + A01) * B10
    # A00 -> U2[0,4]=1
    # A01 -> U2[1,4]=1
    # B10 is j=1,k=0 -> V2[2,4]=1
    # Contributes to C00 (k=0,i=0 -> W2[0,4]) with -1 and C01 (k=1,i=0 -> W2[2,4])
    U2[0][4] = 1
    U2[1][4] = 1
    V2[2][4] = 1
    W2[0][4] = -1
    W2[2][4] = 1
    
    # m6 = (A10 - A00) * (B01 + B11)
    # A10 is i=1,j=0 -> U2[2,5]=1
    # A00 is i=0,j=0 -> U2[0,5]=-1
    # B01 is j=0,k=1 -> V2[1,5]=1
    # B11 is j=1,k=1 -> V2[3,5]=1
    # Contributes to C11 (k=1,i=1 -> W2[3,5])
    U2[2][5] = 1
    U2[0][5] = -1
    V2[1][5] = 1
    V2[3][5] = 1
    W2[3][5] = 1
    
    # m7 = (A01 - A10) * (B00 + B10)
    # A01 is i=0,j=1 -> U2[1,6]=1
    # A10 is i=1,j=0 -> U2[2,6]=-1
    # B00 is j=0,k=0 -> V2[0,6]=1
    # B10 is j=1,k=0 -> V2[2,6]=1
    # Contributes to C00 (k=0,i=0 -> W2[0,6])
    U2[1][6] = 1
    U2[2][6] = -1
    V2[0][6] = 1
    V2[2][6] = 1
    W2[0][6] = 1

    # Block Strassen Coefficients
    # cA[r][I][J], cB[r][J][K], cC[r][K][I]
    # r in 0..6
    
    # r=0 (m1): A00 * (B00+B11)
    # A00: I=0, J=0. cA[0][0][0] = 1
    # B00: J=0, K=0. cB[0][0][0] = 1
    # B11: J=1, K=1. cB[0][1][1] = 1
    # C00: K=0, I=0. cC[0][0][0] = 1
    # C11: K=1, I=1. cC[0][1][1] = 1
    
    cA = [[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    cB = [[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    cC = [[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # r=0
    cA[0][0][0] = 1
    cB[0][0][0] = 1
    cB[0][1][1] = 1
    cC[0][0][0] = 1
    cC[0][1][1] = 1
    
    # r=1 (m2): (A00+A10) * B11
    cA[1][0][0] = 1
    cA[1][1][0] = 1
    cB[1][1][1] = 1
    cC[1][1][0] = 1
    cC[1][1][1] = -1
    
    # r=2 (m3): (A01-A11) * (B00+B01)
    cA[2][0][1] = 1
    cA[2][1][1] = -1
    cB[2][0][0] = 1
    cB[2][0][1] = 1
    cC[2][1][0] = 1
    cC[2][1][1] = 1
    
    # r=3 (m4): A11 * (B10-B00)
    cA[3][1][1] = 1
    cB[3][1][0] = 1
    cB[3][0][0] = -1
    cC[3][0][0] = 1
    cC[3][1][0] = 1
    
    # r=4 (m5): (A00+A01) * B10
    cA[4][0][0] = 1
    cA[4][0][1] = 1
    cB[4][1][0] = 1
    cC[4][0][0] = -1
    cC[4][1][0] = 1
    
    # r=5 (m6): (A10-A00) * (B01+B11)
    cA[5][1][0] = 1
    cA[5][0][0] = -1
    cB[5][0][1] = 1
    cB[5][1][1] = 1
    cC[5][1][1] = 1
    
    # r=6 (m7): (A01-A10) * (B00+B10)
    cA[6][0][1] = 1
    cA[6][1][0] = -1
    cB[6][0][0] = 1
    cB[6][1][0] = 1
    cC[6][0][0] = 1

    # Construct U, V, W
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for I in range(2):
                for J in range(2):
                    ca = cA[r][I][J]
                    if ca != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_val = U2[i_prime * 2 + j_prime][s]
                                if u_val != 0:
                                    i = 2 * I + i_prime
                                    j = 2 * J + j_prime
                                    U[i * 4 + j][col] += ca * u_val
                            
                    for K in range(2):
                        cb = cB[r][J][K]
                        if cb != 0:
                            for j_prime in range(2):
                                for k_prime in range(2):
                                    v_val = V2[j_prime * 2 + k_prime][s]
                                    if v_val != 0:
                                        j = 2 * J + j_prime
                                        k = 2 * K + k_prime
                                        V[j * 4 + k][col] += cb * v_val
                                        
                    for K in range(2):
                        cc = cC[r][K][I]
                        if cc != 0:
                            for k_prime in range(2):
                                for i_prime in range(2):
                                    w_val = W2[k_prime * 2 + i_prime][s]
                                    if w_val != 0:
                                        k = 2 * K + k_prime
                                        i = 2 * I + i_prime
                                        W[k * 4 + i][col] += cc * w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
