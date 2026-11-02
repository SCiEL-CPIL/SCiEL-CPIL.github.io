"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _add_strassen_block(U, V, W, col, b, n, m, p):
    """
    Add 7 rank-1 terms for a 2x2 by 2x2 block multiplication.
    A is rows 0-1, cols 0-1.
    B is rows 0-1, cols 2b, 2b+1.
    C is rows 2b, 2b+1, cols 0-1.
    Returns new col index.
    """
    # A indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B indices: (0, 2b)->2b, (0, 2b+1)->2b+1, (1, 2b)->8+2b, (1, 2b+1)->8+2b+1
    # C indices: (2b, 0)->4b, (2b, 1)->4b+1, (2b+1, 0)->4b+2, (2b+1, 1)->4b+3
    
    k0 = 2 * b
    k1 = 2 * b + 1
    
    # M1 = (A00 + A11) * (B0k0 + B1k1)
    U[0][col] += 1
    U[4][col] += 1
    V[k0][col] += 1
    V[8 + k1][col] += 1
    W[k0 * n + 0][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1

    # M2 = (A10 + A11) * B0k0
    U[3][col] += 1
    U[4][col] += 1
    V[k0][col] += 1
    W[k0 * n + 0][col] += 1
    W[k1 * n + 0][col] += -1
    col += 1

    # M3 = A00 * (B0k1 - B1k1)
    U[0][col] += 1
    V[k1][col] += 1
    V[8 + k1][col] += -1
    W[k1 * n + 0][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1

    # M4 = A11 * (B1k0 - B0k0)
    U[4][col] += 1
    V[8 + k0][col] += 1
    V[k0][col] += -1
    W[k0 * n + 0][col] += 1
    W[k0 * n + 1][col] += 1
    col += 1

    # M5 = (A00 + A01) * B1k1
    U[0][col] += 1
    U[1][col] += 1
    V[8 + k1][col] += 1
    W[k0 * n + 0][col] += -1
    W[k1 * n + 0][col] += 1
    col += 1

    # M6 = (A10 - A00) * (B0k0 + B0k1)
    U[3][col] += 1
    U[0][col] += -1
    V[k0][col] += 1
    V[k1][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1

    # M7 = (A01 - A11) * (B1k0 + B1k1)
    U[1][col] += 1
    U[4][col] += -1
    V[8 + k0][col] += 1
    V[8 + k1][col] += 1
    W[k0 * n + 0][col] += 1
    col += 1

    return col

def construct():
    n, m, p = DIMENSIONS
    # Rank: 16 (for A[:,2]*B[2,:]) + 4*7 (for A[:,:2]*B[:2,:]) = 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 2] * B[2, :]
    # A indices: i in {0,1}, j=2. U row: i*m + 2.
    # B indices: j=2, k in {0..7}. V row: 2*p + k.
    # C indices: i in {0,1}, k in {0..7}. W row: k*n + i.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 0:2] * B[0:2, :]
    # Split B into 4 blocks of width 2.
    for b in range(4):
        col = _add_strassen_block(U, V, W, col, b, n, m, p)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
