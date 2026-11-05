"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 2 * (n * m) = 2 * 9 = 18
    rank = 2 * n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Split B into two 3x3 blocks: B1 (cols 0-2) and B2 (cols 3-5)
    # C1 = A * B1, C2 = A * B2
    # C = [C1, C2]
    
    # Part 1: k in 0..2
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(m): # 0..2
                r = i * m + j * m + k # This mapping is wrong, rank is 18, index needs to be unique per term
                # Actually, term index r should be 0..8 for first part
                pass
    
    # Let's just use a counter
    r = 0
    # First block of columns (0 to m-1)
    for i in range(n):
        for j in range(m):
            for k in range(m):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    # Second block of columns (m to 2*m-1)
    for i in range(n):
        for j in range(m):
            for k in range(m):
                k_actual = m + k
                U[i * m + j][r] = 1
                V[j * p + k_actual][r] = 1
                W[k_actual * n + i][r] = 1
                r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
