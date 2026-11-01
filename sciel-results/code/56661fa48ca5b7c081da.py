"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Rank 32 decomposition: 4 Strassen blocks (2 cols each) + 1 schoolbook column
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    # Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = 2 * block  # start column of this block
        # Strassen's 7 products
        # U vectors (A indices: 0=a00, 1=a01, 2=a10, 3=a11)
        strassen_U = [
            [1, 0, 0, 1],   # M1: a00+a11
            [0, 0, 1, 1],   # M2: a10+a11
            [1, 0, 0, 0],   # M3: a00
            [0, 0, 0, 1],   # M4: a11
            [1, 1, 0, 0],   # M5: a00+a01
            [0, 0, 1, -1],  # M6: a10-a00
            [0, 1, 0, -1],  # M7: a01-a11
        ]
        # V vectors (B indices for this block: k0=b0,k0; k0+1=b0,k0+1; 9+k0=b1,k0; 9+k0+1=b1,k0+1)
        strassen_V = [
            {k0: 1, 9+k0+1: 1},       # M1: b00+b11
            {k0: 1},                   # M2: b00
            {9+k0: 1, 9+k0+1: -1},    # M3: b10-b11
            {k0: -1, k0+1: 1},        # M4: b01-b00
            {9+k0+1: 1},              # M5: b11
            {k0: 1, 9+k0: 1},         # M6: b00+b10
            {k0+1: 1, 9+k0+1: 1},     # M7: b01+b11
        ]
        # W vectors (C indices: 2*k0=c0,k0; 2*k0+1=c1,k0; 2*k0+2=c0,k0+1; 2*k0+3=c1,k0+1)
        strassen_W = [
            {2*k0: 1, 2*k0+3: 1},     # M1: c00+=1, c11+=1
            {2*k0+1: 1, 2*k0+3: -1},  # M2: c10+=1, c11+=-1
            {2*k0+2: 1, 2*k0+3: 1},   # M3: c01+=1, c11+=1
            {2*k0: 1, 2*k0+1: 1},     # M4: c00+=1, c10+=1
            {2*k0: -1, 2*k0+2: 1},    # M5: c00+=-1, c01+=1
            {2*k0+3: 1},              # M6: c11+=1
            {2*k0: 1},                # M7: c00+=1
        ]
        
        for s in range(7):
            for i in range(4):
                U[i][col] = strassen_U[s][i]
            for idx, val in strassen_V[s].items():
                V[idx][col] = val
            for idx, val in strassen_W[s].items():
                W[idx][col] = val
            col += 1
    
    # Schoolbook for column 8
    k8 = 8
    schoolbook = [
        # (A_idx, B_idx, C_idx)
        (0, k8, 2*k8),       # a00*b0,8 → c0,8
        (1, 9+k8, 2*k8),     # a01*b1,8 → c0,8
        (2, k8, 2*k8+1),     # a10*b0,8 → c1,8
        (3, 9+k8, 2*k8+1),   # a11*b1,8 → c1,8
    ]
    for a_idx, b_idx, c_idx in schoolbook:
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[c_idx][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
