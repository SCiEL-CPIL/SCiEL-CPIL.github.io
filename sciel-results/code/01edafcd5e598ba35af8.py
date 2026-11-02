"""Matrix multiplication 2x3 by 3x4 tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22

    # Block 1: cols 0-7, handle k=0,1 for all i,j
    # For each t in 0..7: i = t//4, k = t%4
    # U[i*3+j, t] = 1 for j=0,1,2
    # V[j*4+k, t] = 1 for j=0,1,2
    # W[k*2+i, t] = 1
    for t in range(8):
        i = t // 4
        k = t % 4
        for j in range(3):
            U[i * 3 + j][t] = 1
            V[j * 4 + k][t] = 1
        W[k * 2 + i][t] = 1

    # Block 2: cols 8-21, Strassen on 2x2 blocks for j=1,2 and k=0,1 (instance 2a) and k=2,3 (instance 2b)
    # Strassen 2x2 coefficients
    # U coeffs for A (2x2): rows correspond to a,b,c,d (i.e., positions 0,1,2,3)
    # V coeffs for B (2x2): rows correspond to e,f,g,h
    # W coeffs for C (2x2): rows correspond to c00,c01,c10,c11
    
    # Strassen products M1..M7:
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    
    # U coefficients (for A blocks, 2x2 -> 4 rows):
    # M1: a=1, d=1
    # M2: c=1, d=1
    # M3: a=1
    # M4: d=1
    # M5: a=1, b=1
    # M6: c=1, a=-1
    # M7: b=1, d=-1
    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    
    # V coefficients (for B blocks, 2x2 -> 4 rows):
    # M1: e=1, h=1
    # M2: e=1
    # M3: f=1, h=-1
    # M4: g=1, e=-1
    # M5: h=1
    # M6: e=1, f=1
    # M7: g=1, h=1
    V_strassen = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    
    # W coefficients (for C blocks, 2x2 -> 4 rows):
    # M1: c00=1, c11=1
    # M2: c10=1
    # M3: c01=1, c11=1
    # M4: c00=1, c10=1
    # M5: c00=1, c01=1
    # M6: c11=1
    # M7: c00=1, c11=1
    W_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 1],  # M7
    ]
    
    # Instance 2a: cols 8-14, j=1,2; k=0,1; i=0,1
    # A block: rows i=0,1; cols j=1,2 -> U rows: i*3+1, i*3+2
    # B block: rows j=1,2; cols k=0,1 -> V rows: j*4+k
    # C block: rows k=0,1; cols i=0,1 -> W rows: k*2+i
    
    for s in range(7):
        col = 8 + s
        # U: for i=0,1 and j=1,2
        for i in range(2):
            for j in range(2):
                U[i * 3 + (j + 1)][col] = U_strassen[s][i * 2 + j]
        # V: for j=1,2 and k=0,1
        for j in range(2):
            for k in range(2):
                V[(j + 1) * 4 + k][col] = V_strassen[s][j * 2 + k]
        # W: for k=0,1 and i=0,1
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][col] = W_strassen[s][k * 2 + i]
    
    # Instance 2b: cols 15-21, j=1,2; k=2,3; i=0,1
    # A block: same as 2a
    # B block: rows j=1,2; cols k=2,3 -> V rows: j*4+k
    # C block: rows k=2,3; cols i=0,1 -> W rows: k*2+i
    
    for s in range(7):
        col = 15 + s
        # U: same as 2a
        for i in range(2):
            for j in range(2):
                U[i * 3 + (j + 1)][col] = U_strassen[s][i * 2 + j]
        # V: for j=1,2 and k=2,3
        for j in range(2):
            for k in range(2):
                V[(j + 1) * 4 + (k + 2)][col] = V_strassen[s][j * 2 + k]
        # W: for k=2,3 and i=0,1
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i][col] = W_strassen[s][k * 2 + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
