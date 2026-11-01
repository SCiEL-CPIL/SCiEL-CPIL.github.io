"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=3 into m1=2 and m2=1
    m1 = 2
    m2 = 1
    
    # Part 1: 2 x 2 x 10 (Schoolbook, rank 40)
    rank1 = m1 * n * p
    U1 = [[0] * rank1 for _ in range(n * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p)]
    W1 = [[0] * rank1 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U1[i * m1 + j][col] = 1
                V1[j * p + k][col] = 1
                W1[k * n + i][col] = 1
                col += 1
                
    # Part 2: 2 x 1 x 10 (Rank 1)
    rank2 = 1
    U2 = [[0] * rank2 for _ in range(n * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p)]
    W2 = [[0] * rank2 for _ in range(p * n)]
    
    # T_{i,j,0} = 1 for i in 0..1, j in 0..9, k=0
    # U[i, 0] = 1
    # V[j, 0] = 1
    # W[j, i, 0] = 1
    for i in range(n):
        U2[i * m2 + 0][0] = 1
    for j in range(p):
        V2[0 * p + j][0] = 1
    for j in range(p):
        for i in range(n):
            W2[j * n + i][0] = 1

    # Combine
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill U
    # A is 2x3. Indices: i in 0..1, j in 0..2.
    # U1 corresponds to j in 0..1.
    # U2 corresponds to j=2.
    for i in range(n):
        for j in range(m1):
            for c in range(rank1):
                U[i * m + j][c] = U1[i * m1 + j][c]
        for c in range(rank2):
            U[i * m + m1][rank1 + c] = U2[i * m2 + 0][c]
            
    # Fill V
    # B is 3x10. Indices: j in 0..2, k in 0..9.
    # V1 corresponds to j in 0..1.
    # V2 corresponds to j=2.
    for j in range(m1):
        for k in range(p):
            for c in range(rank1):
                V[j * p + k][c] = V1[j * p + k][c]
    for k in range(p):
        for c in range(rank2):
            V[(m1) * p + k][rank1 + c] = V2[0 * p + k][c]
            
    # Fill W
    # C is 10x2 (flattened k*n+i).
    # W1 and W2 have same structure.
    for k in range(p):
        for i in range(n):
            for c in range(rank1):
                W[k * n + i][c] = W1[k * n + i][c]
            for c in range(rank2):
                W[k * n + i][rank1 + c] = W2[k * n + i][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
