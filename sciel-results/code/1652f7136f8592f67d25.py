# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: Split into 4x4x4 and 4x4x1 parts
    # Rank 4x4x4 schoolbook is 16
    # Rank 4x4x1 schoolbook is 16
    # Total rank 32
    
    rank_total = 32
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Part 1: k in 0..3 (4x4x4)
    for i in range(n):
        for j in range(m):
            t = i * m + j
            U[i * m + j][t] = 1
            for k in range(4):
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                
    # Part 2: k = 4 (4x4x1)
    for i in range(n):
        for j in range(m):
            t = 16 + i * m + j
            U[i * m + j][t] = 1
            V[j * p + 4][t] = 1
            W[4 * n + i][t] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
