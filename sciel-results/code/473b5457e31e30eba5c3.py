"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition m=4 into 1+3, p=5 into 1+4
    # Sub-problems:
    # 1. 2x1x1 (j=0, k=0) -> Rank 2
    # 2. 2x3x1 (j=1..3, k=0) -> Rank 6
    # 3. 2x1x4 (j=0, k=1..4) -> Rank 1
    # 4. 2x3x4 (j=1..3, k=1..4) -> Rank 8
    
    total_rank = 2 + 6 + 1 + 8
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, c):
        U[i * m + j][c] += 1
        V[j * p + k][c] += 1
        W[k * n + i][c] += 1

    # 1. 2x1x1 (j=0, k=0)
    # Terms: (0,0,0) and (1,0,0)
    add_term(0, 0, 0, col); col += 1
    add_term(1, 0, 0, col); col += 1
    
    # 2. 2x3x1 (j=1..3, k=0)
    # Terms: (i,j,0) for i in 0..1, j in 1..3
    for i in range(n):
        for j in range(1, 4):
            add_term(i, j, 0, col); col += 1
            
    # 3. 2x1x4 (j=0, k=1..4)
    # Rank 1: U[i,0]=1, V[0,k]=1, W[k,i]=1 for all i,k?
    # We need U[i, col] V[0, col] W[k, col] = 1 for all i in 0..1, k in 1..4.
    # Set U[i, col] = 1, V[0, col] = 1, W[k, col] = 1.
    for i in range(n):
        U[i * m + 0][col] = 1
    for k in range(1, 5):
        V[0 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # 4. 2x3x4 (j=1..3, k=1..4)
    # Split k into k=1 and k=2..4
    # Part 4a: 2x3x1 (j=1..3, k=1) -> Rank 3 (for i=0) + 3 (for i=1) = 6?
    # Wait, 2x3x1 rank is 6.
    # Part 4b: 2x3x3 (j=1..3, k=2..4) -> Rank 1 (for i=0) + 1 (for i=1) = 2?
    # 1x3x3 rank is 1.
    
    # 4a: j=1..3, k=1
    for i in range(n):
        for j in range(1, 4):
            add_term(i, j, 1, col); col += 1
            
    # 4b: j=1..3, k=2..4
    # For i=0: Rank 1. U[0, j]=1, V[j, k]=1, W[k, 0]=1?
    # We need T[0, j, k] = 1.
    # Set U[0, m_j] = 1, V[j, p_k] = 1, W[k, 0] = 1?
    # This doesn't work directly with add_term.
    # We fill manually.
    
    # For i=0
    for j in range(1, 4):
        U[0 * m + j][col] = 1
    for j in range(1, 4):
        for k in range(2, 5):
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
    col += 1
    
    # For i=1
    for j in range(1, 4):
        U[1 * m + j][col] = 1
    for j in range(1, 4):
        for k in range(2, 5):
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
