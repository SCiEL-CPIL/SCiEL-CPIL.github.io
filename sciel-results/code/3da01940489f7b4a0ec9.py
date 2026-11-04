import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen factors for 2x2
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # c=1: M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # c=2: M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[0][2] = 1; V2[1][2] = -1
    W2[1][2] = 1; W2[3][2] = -1
    
    # c=3: M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # c=4: M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            u_row = i * m + j
            for c1 in range(7):
                for c0 in range(7):
                    c = c1 * 7 + c0
                    val = U2[i1*2+j1][c1] * U2[i0*2+j0][c0]
                    if val != 0:
                        U[u_row][c] = val
                        
    for j in range(4):
        for k in range(4):
            j1, j0 = divmod(j, 2)
            k1, k0 = divmod(k, 2)
            v_row = j * p + k
            for c1 in range(7):
                for c0 in range(7):
                    c = c1 * 7 + c0
                    val = V2[j1*2+k1][c1] * V2[j0*2+k0][c0]
                    if val != 0:
                        V[v_row][c] = val
                        
    for k in range(4):
        for i in range(4):
            k1, k0 = divmod(k, 2)
            i1, i0 = divmod(i, 2)
            w_row = k * n + i
            for c1 in range(7):
                for c0 in range(7):
                    c = c1 * 7 + c0
                    val = W2[k1*2+i1][c1] * W2[k0*2+i0][c0]
                    if val != 0:
                        W[w_row][c] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
