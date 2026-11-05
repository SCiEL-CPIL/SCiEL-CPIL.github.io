"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # Block partitioning: split j (size 4) into 1+3, split k (size 5) into 1+4
    # Four sub-tensors: (2x1x1), (2x3x1), (2x1x4), (2x3x4)
    # Use schoolbook within each block for correctness
    # Ranks: 2 + 6 + 8 + 24 = 40
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 8 x 40
    V = [[0] * rank for _ in range(m * p)]  # 20 x 40
    W = [[0] * rank for _ in range(p * n)]  # 10 x 40
    
    col = 0
    
    # Block 1: i in [0,1], j=0, k=0 -> 2x1x1, rank 2
    for i in range(n):
        U[i * m + 0][col] = 1
        V[0 * p + 0][col] = 1
        W[0 * n + i][col] = 1
        col += 1
    
    # Block 2: i in [0,1], j in [1,2,3], k=0 -> 2x3x1, rank 6
    for i in range(n):
        for j in range(1, 4):
            U[i * m + j][col] = 1
            V[j * p + 0][col] = 1
            W[0 * n + i][col] = 1
            col += 1
    
    # Block 3: i in [0,1], j=0, k in [1,2,3,4] -> 2x1x4, rank 8
    for i in range(n):
        for k in range(1, 5):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 4: i in [0,1], j in [1,2,3], k in [1,2,3,4] -> 2x3x4, rank 24
    for i in range(n):
        for j in range(1, 4):
            for k in range(1, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
