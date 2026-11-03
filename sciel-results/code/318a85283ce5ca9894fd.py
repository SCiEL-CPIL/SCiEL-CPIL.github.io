"""Matrix multiplication 3x4 by 4x8 using rank-88 Strassen-based decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # (3*4, 88)
    V = [[0] * rank for _ in range(m * p)]  # (4*8, 88)
    W = [[0] * rank for _ in range(p * n)]  # (8*3, 88)
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x8) -> C row 0 (1x8)
    # Schoolbook: 4*8 = 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) times B (4x8) -> C rows 1-2 (2x8)
    # Use Strassen: split 2x4 into two 2x2 blocks, 4x8 into eight 2x2 blocks
    # 2 j-blocks x 4 k-blocks = 8 Strassen 2x2x2 decompositions, 7 terms each = 56
    
    # Strassen 2x2x2 factors
    # A local: [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
    # B local: [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
    # C local: [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    for jb in range(2):
        for kb in range(4):
            for t in range(7):
                # U factors: A rows 1,2; cols jb*2, jb*2+1
                for l in range(4):
                    val = S_U[t][l]
                    if val != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = jb * 2 + lj
                        U[gi * m + gj][col] += val
                # V factors: B rows jb*2, jb*2+1; cols kb*2, kb*2+1
                for l in range(4):
                    val = S_V[t][l]
                    if val != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = jb * 2 + lj
                        gk = kb * 2 + lk
                        V[gj * p + gk][col] += val
                # W factors: C rows 1,2; cols kb*2, kb*2+1
                for l in range(4):
                    val = S_W[t][l]
                    if val != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = kb * 2 + lk
                        W[gk * n + gi][col] += val
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
