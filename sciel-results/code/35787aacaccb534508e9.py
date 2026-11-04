"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # 5 blocks of 2x2x2 (Strassen, rank 7 each) -> 35
    # 1 block of 2x1x10 (Schoolbook, rank 20) -> 20
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # u, v, w are lists of 7 vectors of length 4
    # Indices: i*2+j, j*2+k, k*2+i
    u_strassen = [
        [1, 0, 1, 0], # p1
        [0, 0, 1, 1], # p2
        [1, 0, 0, 0], # p3
        [0, 0, 0, 1], # p4
        [1, 1, 0, 0], # p5
        [-1, 0, 1, 0],# p6
        [0, 1, 0, -1] # p7
    ]
    v_strassen = [
        [1, 0, 0, 1], # p1
        [1, 0, 0, 0], # p2
        [0, 1, 0, -1],# p3
        [0, 0, 1, -1],# p4
        [0, 0, 0, 1], # p5
        [1, 1, 0, 0], # p6
        [0, 0, 1, 1]  # p7
    ]
    w_strassen = [
        [1, -1, 1, 1], # p1
        [0, 1, 0, -1],  # p2
        [0, 1, 1, 0],   # p3
        [-1, 1, 0, 1],  # p4
        [1, 0, -1, 0],  # p5
        [0, 0, 0, 1],   # p6
        [1, 0, -1, 0]   # p7
    ]
    
    col = 0
    # 5 blocks of 2x2x2
    # j in {0, 1}, k in {0..9}
    # Block b uses k in {2b, 2b+1}
    for b in range(5):
        for t in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = u_strassen[t][i * 2 + j]
            # V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * p + k_global][col] = v_strassen[t][j * 2 + k_local]
            # W
            for k_local in range(2):
                k_global = 2 * b + k_local
                for i in range(2):
                    W[k_global * n + i][col] = w_strassen[t][k_local * 2 + i]
            col += 1
            
    # 1 block of 2x1x10 (j=2)
    # i in {0, 1}, k in {0..9}
    for k in range(10):
        for i in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
