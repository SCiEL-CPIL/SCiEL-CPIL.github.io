"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # Split m=4 into two blocks of 2
    n_blocks = m // 2  # 2 blocks
    block_size = 2  # each block has 2 columns/rows
    rank = n_blocks * block_size * p  # 2 * 2 * 5 = 20
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 20)
    V = [[0] * rank for _ in range(m * p)]  # (20, 20)
    W = [[0] * rank for _ in range(p * n)]  # (10, 20)
    
    for a in range(n_blocks):  # a = 0, 1
        for bp in range(block_size):  # bp = 0, 1 (local j within block)
            for kp in range(p):  # kp = 0..4
                c = a * block_size * p + bp * p + kp
                for i in range(n):  # i = 0, 1
                    j_global = a * block_size + bp
                    U[i * m + j_global][c] = 1
                    V[j_global * p + kp][c] = 1
                    W[kp * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
