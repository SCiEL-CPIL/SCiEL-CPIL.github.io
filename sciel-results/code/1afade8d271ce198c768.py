"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    # Part 1: 2x3x1 (Rank 6)
    # Part 2: 2x3x2 (Rank 11) = 2x2x2 (Rank 7) + 2x1x2 (Rank 4)
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: k=0
    # C[i, 0] = sum_j A[i, j] B[j, 0]
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 0][col] = 1
            W[0 * n + i][col] = 1
            col += 1
            
    # Part 2: k=1, 2
    # Split A into A1 (cols 0,1) and A2 (col 2)
    # Split B into B11 (rows 0,1 cols 1,2) and B21 (row 2 cols 1,2)
    
    # Sub-part 2a: A1 * B11 (2x2x2 Strassen)
    # A1 indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B11 indices: (0,1)->1, (0,2)->2, (1,1)->4, (1,2)->5
    # W indices: (1,0)->2, (2,0)->4, (1,1)->3, (2,1)->5
    
    # M1 = (A00+A11)(B01+B12)
    U[0][col] = 1; U[4][col] = 1
    V[1][col] = 1; V[5][col] = 1
    W[2][col] = 1; W[5][col] = 1
    col += 1
    
    # M2 = (A10+A11)B01
    U[3][col] = 1; U[4][col] = 1
    V[1][col] = 1
    W[3][col] = 1; W[5][col] = -1
    col += 1
    
    # M3 = A00(B02-B11)
    U[0][col] = 1
    V[2][col] = 1; V[4][col] = -1
    W[4][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A11(B11-B01)
    U[4][col] = 1
    V[4][col] = 1; V[1][col] = -1
    W[2][col] = -1; W[4][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[2][col] = 1; W[5][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B01+B02)
    U[3][col] = 1; U[0][col] = -1
    V[1][col] = 1; V[2][col] = 1
    W[5][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B11+B12)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[2][col] = 1; W[5][col] = 1
    col += 1
    
    # Sub-part 2b: A2 * B21 (2x1x2)
    # A2 indices: (0,2)->2, (1,2)->5
    # B21 indices: (2,1)->7, (2,2)->8
    
    # N1 = A02 * B21
    U[2][col] = 1
    V[7][col] = 1
    W[2][col] = 1
    col += 1
    
    # N2 = A02 * B22
    U[2][col] = 1
    V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # N3 = A12 * B21
    U[5][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    
    # N4 = A12 * B22
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
