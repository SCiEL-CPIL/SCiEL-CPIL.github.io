"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 3x4x4 into 4 blocks of 3x2x2.
    # Each 3x2x2 block has rank 3.
    # Total rank = 4 * 3 = 12.
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define blocks for m (j index) and p (k index)
    # J blocks: [0, 1], [2, 3]
    # K blocks: [0, 1], [2, 3]
    j_blocks = [[0, 1], [2, 3]]
    k_blocks = [[0, 1], [2, 3]]
    
    col = 0
    for jb_idx in range(2):
        for kb_idx in range(2):
            js = j_blocks[jb_idx]
            ks = k_blocks[kb_idx]
            for i in range(n):
                # s is the column index
                for j in js:
                    # U[i*m + j, s] = 1
                    U[i * m + j][col] = 1
                for j in js:
                    for k in ks:
                        # V[j*p + k, s] = 1
                        V[j * p + k][col] = 1
                for k in ks:
                    # W[k*n + i, s] = 1
                    W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
