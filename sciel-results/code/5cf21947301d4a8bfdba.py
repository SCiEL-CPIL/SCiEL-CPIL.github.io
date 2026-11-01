"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition
    # 3 blocks of 2x2x2 (Strassen rank 7) + 1 block of 2x2x1 (Schoolbook rank 4)
    rank = 3 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # U_s[i, j], V_s[j, k], W_s[k, i]
    # Indices i, j, k in {0, 1}
    
    # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, 1]] # W[k,i]
    
    # M2: (a10+a11)b00 -> c10+=1, c11-=1
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 1], [0, -1]] # W[0,1]=1, W[1,1]=-1
    
    # M3: a00(b10-b11) -> c01+=1, c11+=1
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 0], [1, -1]]
    W3 = [[0, 0], [1, 1]] # W[1,0]=1, W[1,1]=1
    
    # M4: a11(b01-b00) -> c00+=1, c10+=1
    U4 = [[0, 0], [0, 1]]
    V4 = [[-1, 1], [0, 0]]
    W4 = [[1, 1], [0, 0]] # W[0,0]=1, W[0,1]=1
    
    # M5: (a00+a01)b11 -> c00-=1, c01+=1
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[-1, 0], [1, 0]] # W[0,0]=-1, W[1,0]=1
    
    # M6: (a10-a00)(b00+b01) -> c11+=1
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [[0, 0], [0, 1]] # W[1,1]=1
    
    # M7: (a01-a11)(b10+b11) -> c00+=1, c01-=1
    U7 = [[0, 1], [-1, 0]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[1, 0], [-1, 0]] # W[0,0]=1, W[1,0]=-1
    
    strassen_blocks = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, W7[0], W7[1]) # Wait, W7 is a matrix.
    ]
    
    # Correct list structure
    strassen_factors = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    for t in range(3):
        for s in range(7):
            Us, Vs, Ws = strassen_factors[s]
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = Us[i][j]
            for j in range(2):
                for k_loc in range(2):
                    k_glob = 2 * t + k_loc
                    V[j * p + k_glob][col] = Vs[j][k_loc]
            for k_loc in range(2):
                for i in range(2):
                    k_glob = 2 * t + k_loc
                    W[k_glob * n + i][col] = Ws[k_loc][i]
            col += 1
            
    # Last block: k=6 (2x2x1 schoolbook)
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
