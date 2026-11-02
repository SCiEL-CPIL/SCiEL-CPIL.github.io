"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 blocks * 7 (Strassen) = 35
    rank = 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # alpha[s][i*2+j]
    alpha = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, -1, 0] # P7
    ]
    
    # beta[s][j*2+k']
    beta = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # gamma[s][i*2+k']
    gamma = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2 # 5 blocks
    
    for t in range(num_blocks):
        for s in range(7):
            r = t * 7 + s
            
            # Fill U
            # U index: i*m + j = 2i + j
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = alpha[s][i * m + j]
            
            # Fill V
            # V index: j*p + k = 10j + k
            # k = 2t + k'
            for j in range(m):
                for kp in range(2):
                    k_global = 2 * t + kp
                    V[j * p + k_global][r] = beta[s][j * 2 + kp]
            
            # Fill W
            # W index: k*n + i = 2k + i
            # k = 2t + k'
            for i in range(n):
                for kp in range(2):
                    k_global = 2 * t + kp
                    W[k_global * n + i][r] = gamma[s][i * 2 + kp]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
