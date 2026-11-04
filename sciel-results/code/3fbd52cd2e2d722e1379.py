"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2
    # U[I, J], V[J, K], W[K, I]
    # c=0: M1
    U0 = [[1, 0], [0, 1]]
    V0 = [[1, 0], [0, 1]]
    W0 = [[1, 0], [0, 1]] # W[K, I] -> W[0,0]=1, W[1,1]=1
    
    # c=1: M2
    U1 = [[0, 0], [1, 1]]
    V1 = [[1, 0], [0, 0]]
    W1 = [[0, 1], [0, 1]] # W[0,1]=1 (C10), W[1,1]=1 (C11)
    
    # c=2: M3
    U2 = [[1, 0], [0, 0]]
    V2 = [[0, 0], [1, -1]]
    W2 = [[0, 1], [1, 1]] # W[0,1]=1 (C01), W[1,1]=1 (C11)
    
    # c=3: M4
    U3 = [[0, 0], [0, 1]]
    V3 = [[-1, 1], [0, 0]]
    W3 = [[1, 0], [1, 0]] # W[0,0]=1 (C00), W[1,0]=1 (C10)
    
    # c=4: M5
    U4 = [[1, 1], [0, 0]]
    V4 = [[0, 0], [0, 1]]
    W4 = [[-1, 0], [1, 0]] # W[0,0]=-1 (C00), W[1,0]=1 (C01? No, C01 is W[1,0] in W[K,I] notation? 
                            # C_IK. K=1, I=0 -> W[1,0]. Yes.)
    
    # c=5: M6
    U5 = [[-1, 0], [1, 0]]
    V5 = [[1, 1], [0, 0]]
    W5 = [[0, 0], [0, 1]] # W[1,1]=1 (C11)
    
    # c=6: M7
    U6 = [[0, 1], [-1, 0]]
    V6 = [[0, 0], [1, 1]]
    W6 = [[1, 0], [0, 0]] # W[0,0]=1 (C00)
    
    factors = [
        (U0, V0, W0),
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6)
    ]
    
    for c in range(7):
        Uc, Vc, Wc = factors[c]
        for d in range(7):
            Ud, Vd, Wd = factors[d]
            r = 7 * c + d
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    u_val = Uc[I][J] * Ud[i][j]
                                    if u_val != 0:
                                        row_u = (2 * I + i) * 4 + (2 * J + j)
                                        U[row_u][r] += u_val
                                    
                                    v_val = Vc[J][K] * Vd[j][k]
                                    if v_val != 0:
                                        row_v = (2 * J + j) * 4 + (2 * K + k)
                                        V[row_v][r] += v_val
                                        
                                    w_val = Wc[K][I] * Wd[k][i]
                                    if w_val != 0:
                                        row_w = (2 * K + k) * 4 + (2 * I + i)
                                        W[row_w][r] += w_val
                                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
