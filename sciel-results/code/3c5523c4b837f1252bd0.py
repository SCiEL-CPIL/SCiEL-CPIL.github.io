"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base Strassen 2x2x2 vectors
    # u, v, w are lists of 7 vectors, each of length 4
    u_base = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    v_base = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    w_base = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]

    rank = 49 + 49 + 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]

    def add_strassen_block(u_offset, v_offset, w_offset, col_offset):
        # u_offset: row offset in U (0)
        # v_offset: row offset in V (0 for first 4x4, 36 for second? No, V is 4x9.
        # V rows: j*9 + k.
        # For first 4x4 (k=0..3): rows 0..3, 9..12, 18..21, 27..30.
        # For second 4x4 (k=4..7): rows 4..7, 13..16, 22..25, 31..34.
        # For third 4x1 (k=8): rows 8, 17, 26, 35.
        
        for s in range(7):
            for r in range(7):
                col = col_offset + s * 7 + r
                u_c = u_base[s][r]
                v_c = v_base[s][r]
                w_c = w_base[s][r]
                
                # Expand u_c (coeff for A block)
                if u_c != 0:
                    bi, bj = r // 2, r % 2
                    for di in range(2):
                        for dj in range(2):
                            gi = 2*bi + di
                            gj = 2*bj + dj
                            U[gi*4 + gj][col] += u_c
                            
                # Expand v_c (coeff for B block)
                if v_c != 0:
                    bi, bj = r // 2, r % 2
                    for di in range(2):
                        for dj in range(2):
                            gi = 2*bi + di
                            gj = 2*bj + dj
                            # B row = gi, B col = gj + v_offset
                            v_row = gi*9 + (gj + v_offset)
                            V[v_row][col] += v_c
                            
                # Expand w_c (coeff for C block)
                if w_c != 0:
                    bi, bj = r // 2, r % 2
                    for di in range(2):
                        for dj in range(2):
                            gi = 2*bi + di
                            gj = 2*bj + dj
                            # C row = gj + w_offset, C col = gi
                            # W index: k*n + i. k is row of C, i is col of C.
                            w_row = (gj + w_offset)*4 + gi
                            W[w_row][col] += w_c

    # Part 1: 4x4x4, B cols 0-3, C cols 0-3
    add_strassen_block(0, 0, 0, 0)
    
    # Part 2: 4x4x4, B cols 4-7, C cols 4-7
    add_strassen_block(0, 4, 4, 49)
    
    # Part 3: 4x4x1, B col 8, C col 8
    # Trivial
    col = 98
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            # B row j, B col 8
            V[j*9 + 8][col] = 1
            # C row 8, C col i
            W[8*4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
