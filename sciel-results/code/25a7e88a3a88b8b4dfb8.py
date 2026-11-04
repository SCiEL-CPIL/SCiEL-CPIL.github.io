"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 39
    # Part 1: 2x1x7 (Rank 14)
    # Part 2: 2x2x7 (Rank 25) -> 3x(2x2x2 Rank 7) + 1x(2x2x1 Rank 4)
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x7
    # j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: 2x2x7
    # j in {1, 2}
    # k blocks: (0,1), (2,3), (4,5) are 2x2x2. (6) is 2x2x1.
    
    # Helper to add a 2x2x2 block using Strassen (Rank 7)
    def add_strassen_block(j1, j2, k1, k2, start_col):
        # a, b are A[0, j1], A[0, j2]
        # c, d are A[1, j1], A[1, j2]
        # e, f are B[j1, k1], B[j1, k2]
        # g, h are B[j2, k1], B[j2, k2]
        
        # M1 = (a+b)(e+f)
        U[0 * m + j1][start_col] += 1
        U[0 * m + j2][start_col] += 1
        V[j1 * p + k1][start_col] += 1
        V[j1 * p + k2][start_col] += 1
        # W: C0k1, C0k2, C1k1, C1k2
        # C0k1 += M1? No, reconstruction handles W.
        # We just define the U and V parts for the 7 products.
        # W is filled during reconstruction.
        
        # M2 = (c+d)e
        U[1 * m + j1][start_col + 1] += 1
        U[1 * m + j2][start_col + 1] += 1
        V[j1 * p + k1][start_col + 1] += 1
        
        # M3 = af
        U[0 * m + j1][start_col + 2] += 1
        V[j1 * p + k2][start_col + 2] += 1
        
        # M4 = dg
        U[1 * m + j2][start_col + 3] += 1
        V[j2 * p + k1][start_col + 3] += 1
        
        # M5 = (a+d)(e+g)
        U[0 * m + j1][start_col + 4] += 1
        U[1 * m + j2][start_col + 4] += 1
        V[j1 * p + k1][start_col + 4] += 1
        V[j2 * p + k1][start_col + 4] += 1
        
        # M6 = (b+c)(f+h)
        U[0 * m + j2][start_col + 5] += 1
        U[1 * m + j1][start_col + 5] += 1
        V[j1 * p + k2][start_col + 5] += 1
        V[j2 * p + k2][start_col + 5] += 1
        
        # M7 = (b+d)(h-g)
        U[0 * m + j2][start_col + 6] += 1
        U[1 * m + j2][start_col + 6] += 1
        V[j2 * p + k2][start_col + 6] += 1
        V[j2 * p + k1][start_col + 6] -= 1
        
        # Now fill W for this block
        # C0k1 = M5 + M4 - M2 - M7
        W[k1 * n + 0][start_col + 4] += 1
        W[k1 * n + 0][start_col + 3] += 1
        W[k1 * n + 0][start_col + 1] -= 1
        W[k1 * n + 0][start_col + 6] -= 1
        
        # C0k2 = M1 + M3 - M4 - M5
        W[k2 * n + 0][start_col] += 1
        W[k2 * n + 0][start_col + 2] += 1
        W[k2 * n + 0][start_col + 3] -= 1
        W[k2 * n + 0][start_col + 4] -= 1
        
        # C1k1 = M2 + M7 - M4 - M5
        W[k1 * n + 1][start_col + 1] += 1
        W[k1 * n + 1][start_col + 6] += 1
        W[k1 * n + 1][start_col + 3] -= 1
        W[k1 * n + 1][start_col + 4] -= 1
        
        # C1k2 = M1 + M6 - M2 - M3
        W[k2 * n + 1][start_col] += 1
        W[k2 * n + 1][start_col + 5] += 1
        W[k2 * n + 1][start_col + 1] -= 1
        W[k2 * n + 1][start_col + 2] -= 1
        
        return start_col + 7

    # Add 3 Strassen blocks
    col = add_strassen_block(1, 2, 0, 1, col)
    col = add_strassen_block(1, 2, 2, 3, col)
    col = add_strassen_block(1, 2, 4, 5, col)
    
    # Add 1 schoolbook block for k=6 (2x2x1 Rank 4)
    # j1=1, j2=2, k1=6
    j1, j2, k1 = 1, 2, 6
    # ae
    U[0 * m + j1][col] += 1
    V[j1 * p + k1][col] += 1
    W[k1 * n + 0][col] += 1
    col += 1
    # bg
    U[0 * m + j2][col] += 1
    V[j2 * p + k1][col] += 1
    W[k1 * n + 0][col] += 1
    col += 1
    # ce
    U[1 * m + j1][col] += 1
    V[j1 * p + k1][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    # dg
    U[1 * m + j2][col] += 1
    V[j2 * p + k1][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
