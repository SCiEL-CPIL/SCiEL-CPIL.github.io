"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 124 (7 for Strassen 2x2 + 117 for schoolbook others)
    rank = 124
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for the 2x2x2 block (indices 0,1)
    # U_vec, V_vec, W_vec for each of the 7 multiplications
    # Indices in 2x2 block: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    
    strassen_U = [
        [1, 0, 0, 1],  # r=0: a+d
        [0, 0, 1, 1],  # r=1: c+d
        [1, 0, 0, 0],  # r=2: a
        [0, 0, 0, 1],  # r=3: d
        [1, 1, 0, 0],  # r=4: a+b
        [-1, 0, 1, 0], # r=5: c-a
        [0, 1, 0, -1]  # r=6: b-d
    ]
    
    strassen_V = [
        [1, 1, 1, 1],  # r=0: e+h
        [1, 0, 0, 0],  # r=1: e
        [0, 1, 0, -1], # r=2: f-h
        [-1, 0, 1, 0], # r=3: g-e
        [0, 0, 0, 1],  # r=4: h
        [1, 1, 0, 0],  # r=5: e+f
        [0, 0, 1, 1]   # r=6: g+h
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # r=0: C00 += 1, C11 += 1
        [0, 0, 1, -1], # r=1: C10 += 1, C11 -= 1
        [0, 1, 0, 1],  # r=2: C01 += 1, C11 += 1
        [1, 0, 1, 0],  # r=3: C00 += 1, C10 += 1
        [-1, 1, 0, 0], # r=4: C00 -= 1, C01 += 1
        [0, 0, 0, 1],  # r=5: C11 += 1
        [1, 0, 0, 0]   # r=6: C00 += 1
    ]
    
    # Map 2x2 indices to 5x5 indices
    # 2x2 index idx -> (i, j)
    # 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
    def map_2x2_to_5x5(idx):
        if idx == 0: return 0 * 5 + 0
        if idx == 1: return 0 * 5 + 1
        if idx == 2: return 1 * 5 + 0
        if idx == 3: return 1 * 5 + 1
        return 0

    # Add Strassen terms to columns 0-6
    for r in range(7):
        for idx in range(4):
            val_u = strassen_U[r][idx]
            val_v = strassen_V[r][idx]
            val_w = strassen_W[r][idx]
            
            if val_u != 0:
                U[map_2x2_to_5x5(idx)][r] = val_u
            if val_v != 0:
                V[map_2x2_to_5x5(idx)][r] = val_v
            if val_w != 0:
                W[map_2x2_to_5x5(idx)][r] = val_w

    # Add Schoolbook terms for all other (i,j,k)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in the 2x2 block
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
